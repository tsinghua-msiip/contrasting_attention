
import hydra
from omegaconf import DictConfig
import pickle


@hydra.main(version_base=None, config_path="config")
def main(config: DictConfig) -> None:

    with open(f'{config.attn_dump_large}.pkl', 'rb') as f:
        attn_large = pickle.load(f)
    with open(f'{config.attn_dump_small}.pkl', 'rb') as f:
        attn_small = pickle.load(f)
    
    # calculate the difference in attention weights
    attn_diff = []
    for eg_attn_large, eg_attn_small in zip(attn_large, attn_small):
        diff = eg_attn_large - eg_attn_small
        attn_diff.append(diff)
    
    with open(f'{config.attn_diff}.pkl', 'wb') as f:
        pickle.dump(attn_diff, f)


if __name__ == "__main__":
    main()