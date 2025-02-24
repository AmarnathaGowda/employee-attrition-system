# ml_engine/management/commands/train_models.py
from django.core.management.base import BaseCommand
from ml_engine.train import train_models

class Command(BaseCommand):
    help = 'Train and save ML models'

    def handle(self, *args, **kwargs):
        results = train_models()
        for model, metrics in results.items():
            self.stdout.write(self.style.SUCCESS(
                f"{model}: "
                f"Accuracy={metrics['accuracy']:.3f}, "
                f"ROC AUC={metrics['roc_auc']:.3f}"
            ))

