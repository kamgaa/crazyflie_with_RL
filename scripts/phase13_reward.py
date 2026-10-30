"""Opt-in phase13: change only Z velocity cost to -weight*v_world_z**4."""
import torch
from phase11_reward import TERMS, desired_velocity, euler_zyx
from phase11_reward import components as baseline_components
from phase9_reward import components as legacy_components

def components(o, failed, cfg, velocity_mode=None):
    assert cfg['vertical_velocity_mode']=='absolute_quartic'
    terms=baseline_components(o,failed,cfg)
    terms[:,5]=-cfg['axis_weights']['velocity'][2]*o[:,5].pow(4)
    return terms

def apply_reward(raw,cfg):
    assert cfg['reward_id']=='phase13_axis_vz_quartic' and not cfg['dt_multiplier']
    raw._phase9_reward_config=cfg
    def adjusted():
        o=raw.policy_observation();f=raw.reset_terminated
        raw._phase9_common_stop=legacy_components(o,f,cfg,'stop').sum(-1)
        raw._phase9_common_return=legacy_components(o,f,cfg,'return').sum(-1)
        return components(o,f,cfg).sum(-1)
    raw._get_rewards=adjusted
