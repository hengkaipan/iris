import random
from typing import List, Optional, Union

import gym
from einops import rearrange
import numpy as np
from PIL import Image
import torch
from torch.distributions.categorical import Categorical
import torchvision


class WorldModelPlanner:

    def __init__(self, tokenizer: torch.nn.Module, world_model: torch.nn.Module, device: Union[str, torch.device]) -> None:

        self.device = torch.device(device)
        self.world_model = world_model.to(self.device).eval()
        self.tokenizer = tokenizer.to(self.device).eval()

        self.keys_values_wm, self.obs_tokens, self._num_observations_tokens = None, None, None
        
        self.encoder = 1
        self.decoder = 1


    @property
    def num_observations_tokens(self) -> int:
        return self._num_observations_tokens

    @torch.no_grad()
    def reset_from_initial_observations(self, observations: torch.FloatTensor) -> torch.FloatTensor:
        # if observations is a dict, get only the 'visual' key
        obs_tokens = self.tokenizer.encode(observations, should_preprocess=True).tokens    # (B, C, H, W) -> (B, K)
        _, num_observations_tokens = obs_tokens.shape
        if self.num_observations_tokens is None:
            self._num_observations_tokens = num_observations_tokens

        _ = self.refresh_keys_values_with_initial_obs_tokens(obs_tokens)
        self.obs_tokens = obs_tokens

        return self.decode_obs_tokens()

    @torch.no_grad()
    def refresh_keys_values_with_initial_obs_tokens(self, obs_tokens: torch.LongTensor) -> torch.FloatTensor:
        n, num_observations_tokens = obs_tokens.shape
        assert num_observations_tokens == self.num_observations_tokens
        self.keys_values_wm = self.world_model.transformer.generate_empty_keys_values(n=n, max_tokens=self.world_model.config.max_tokens)
        outputs_wm = self.world_model(obs_tokens, past_keys_values=self.keys_values_wm)
        return outputs_wm.output_sequence  # (B, K, E)

    @torch.no_grad()
    def step(self, action: Union[int, np.ndarray, torch.LongTensor], should_predict_next_obs: bool = True) -> None:
        assert self.keys_values_wm is not None and self.num_observations_tokens is not None

        num_passes = 1 + self.num_observations_tokens if should_predict_next_obs else 1

        output_sequence, obs_tokens = [], []

        if self.keys_values_wm.size + num_passes > self.world_model.config.max_tokens:
            _ = self.refresh_keys_values_with_initial_obs_tokens(self.obs_tokens)

        token = action.clone().detach() if isinstance(action, torch.Tensor) else torch.tensor(action, dtype=torch.long)
        # token = token.reshape(-1, 1).to(self.device)  # (B, 1)
        token = token.to(self.device)

        for k in range(num_passes):  # assumption that there is only one action token.

            outputs_wm = self.world_model(token, past_keys_values=self.keys_values_wm)
            output_sequence.append(outputs_wm.output_sequence)

            if k < self.num_observations_tokens:
                token = Categorical(logits=outputs_wm.logits_observations).sample()
                obs_tokens.append(token)

        output_sequence = torch.cat(output_sequence, dim=1)   # (B, 1 + K, E)
        self.obs_tokens = torch.cat(obs_tokens, dim=1)        # (B, K)

        obs,z,e = self.decode_obs_tokens() if should_predict_next_obs else None
        return obs,z,e 

    @torch.no_grad()
    def render_batch(self) -> List[Image.Image]:
        frames = self.decode_obs_tokens().detach().cpu()
        frames = rearrange(frames, 'b c h w -> b h w c').mul(255).numpy().astype(np.uint8)
        return [Image.fromarray(frame) for frame in frames]

    @torch.no_grad()
    def decode_obs_tokens(self) -> List[Image.Image]:
        embedded_tokens = self.tokenizer.embedding(self.obs_tokens)     # (B, K, E)
        z = rearrange(embedded_tokens, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
        rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
        return torch.clamp(rec, 0, 1), z, embedded_tokens

    @torch.no_grad()
    def render(self):
        assert self.obs_tokens.shape == (1, self.num_observations_tokens)
        return self.render_batch()[0]
    
    @torch.no_grad()
    def decode_obs(self, z_obs):
        """
        input :   z: (b, num_frames, num_patches, emb_dim)
        output: obs: (b, num_frames, 3, img_size, img_size)
        """
        if isinstance(z_obs, dict):
            z_obs = z_obs['visual']
        results = []
        for i in range(z_obs.size(1)):
            z = z_obs[:, i, :, :]
            z = rearrange(z, 'b (h w) e -> b e h w', h=int(np.sqrt(self.num_observations_tokens)))
            rec = self.tokenizer.decode(z, should_postprocess=True)         # (B, C, H, W)
            rec = torch.clamp(rec, 0, 1)
            results.append(rec)
        returned_obs = torch.stack(results, dim=1)
        returned_obs = {'visual': returned_obs}
        return [returned_obs]

    @torch.no_grad()
    def encode_obs(self, obs):
        """
        input : obs (dict): "visual", "proprio" (b, t, 3, img_size, img_size)
        output:   z (dict): "visual", "proprio" (b, t, num_patches, encoder_emb_dim)
        """
        visual = obs['visual']
        b = visual.shape[0]
        results = []
        for i in range(b):
            cur_obs = visual[i]
            obs_tokens = self.tokenizer.encode(cur_obs,should_preprocess=True).tokens
            embedded_tokens = self.tokenizer.embedding(obs_tokens) 
            results.append(embedded_tokens)
        z = torch.stack(results, dim=0)
        return {'visual': z}
    
    @torch.no_grad()
    def rollout(self, obs_0, act, step_size=1):
        """
        input:  obs_0 (dict): (b, n, 3, img_size, img_size)
                  act: (b, t+n, action_dim)
        output: embeddings of rollout obs
                visuals: (b, t+n+1, 3, img_size, img_size)
                z: (b, t+n+1, num_patches, emb_dim)
        """
        if isinstance(obs_0, dict):
            obs_0 = obs_0['visual'].squeeze(1)
        cur_obs, cur_z, cur_e = self.reset_from_initial_observations(obs_0)
        all_e = [cur_e]
        all_obs = [cur_obs]
        for i in range(act.size(1)):
            a = act[:, i, :]
            cur_obs, cur_z, cur_e = self.step(a)
            all_e.append(cur_e)
            all_obs.append(cur_obs)
        z = torch.stack(all_e, dim=1)
        obs = torch.stack(all_obs, dim=1)
        z = {'visual': z}
        z_obses = z
        return z_obses, z
