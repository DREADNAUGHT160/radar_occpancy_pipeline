import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from models.resnet_backbone import RadarResNet


class ModelFactory:
    @staticmethod
    def get_model(config):
        name = config['model']['name']
        if name == 'resnet18_radar':
            return RadarResNet(
                in_channels   = config['model']['in_channels'],
                num_classes   = config['model']['num_classes'],
                doppler_depth = config['model'].get('doppler_depth', 128),
            )
        raise ValueError(f"Unknown model: {name}")
