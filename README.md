# Open Door

A custom task environment for training humanoid robots to open doors using the active-adaptation framework.（目前只实现了到指定位置和方向）

## Project Structure

```
open-door/
├── cfg/
│   └── task/              # Task configuration files
├── src/
│   ├── open_door/         # Environment implementation
│   └── open_door_learning/  # Learning-specific components
├── pyproject.toml
└── README.md
```

## Installation

### Prerequisites
- [active-adaptation](https://github.com/your-repo/active-adaptation) framework installed
- Python 3.11

### Setup

1. **Place the project in the correct directory structure**
   ```bash
   # Ensure open-door is at the same level as active-adaptation
   project/
   ├── active-adaptation/
   └── open-door/  # <- this repository
   ```

2. **Install the package**
   ```bash
   cd open-door
   pip install -e .
   ```

3. **Register with active-adaptation**
   
   Add the following entries to `active-adaptation/.cache/projects.json`:

   ```json
   {
     "environment": {
       "open_door": {
         "value": "open_door",
         "path": "/path/to/your/project/open-door/src/open_door",
         "type": "environment",
         "enabled": true,
         "task_dir": "/path/to/your/project/open-door/cfg/task"
       }
     },
     "learning": {
       "open_door_learning": {
         "value": "open_door_learning",
         "path": "/path/to/your/project/open-door/src/open_door_learning",
         "type": "learning",
         "enabled": true
       }
     }
   }
   ```

   **Note:** Replace `/path/to/your/project` with your actual project directory path.

## Usage

### Training

Train a policy using PPO with symmetry augmentation:（似乎还有问题，训练到一定步数会报错）

```bash
python active-adaptation/scripts/train_ppo.py task=open_door algo=ppo_symaug
```

### Evaluation (Play)

Evaluate a trained policy:

```bash
python active-adaptation/scripts/play.py \
  task=open_door \
  algo=ppo_symaug \
  checkpoint_path=/path/to/checkpoint_final.pt
```

Example with specific checkpoint:
```bash
python active-adaptation/scripts/play.py \
  task=open_door \
  algo=ppo_symaug \
  checkpoint_path=outputs/2026-03-06/23-28-44-G1LocoFlat-ppo_symaug/wandb/run-20260306_232932-1lr1am7e/files/checkpoint_final.pt
```

## Configuration

Task configurations are located in `cfg/task/`. Modify these files to adjust:
- Environment parameters
- Reward functions
- Observation spaces
- Randomization settings

## License

[Add your license information here]