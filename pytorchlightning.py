#keeping computational code
import lightning as L
import torch
import torch.nn as nn
import torch.nn.functional as F


class LitModel(nn.Module):
    def __init__(self):
        super().__init__()
        self.layer_1 = nn.Linear(28 * 28, 128)
        self.layer_2 = nn.Linear(128, 10)

    def forward(self, x):
        x = x.view(x.size(0), -1)
        x = self.layer_1(x)
        x = F.relu(x)
        x = self.layer_2(x)
        return x
    #Configure training logic
class LitModel(L.LightningModule):
    def __init__(self, encoder):
        super().__init__()
        self.encoder = encoder

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.encoder(x)
        loss = F.cross_entropy(y_hat, y)
        return loss
    

    #move optimizer(s) and LR schedulers to the LightningModule
class LitModel(L.LightningModule):
    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.encoder.parameters(), lr=1e-3)
        lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=1)
        return [optimizer], [lr_scheduler]


    #Organize Validation logic
class LitModel(L.LightningModule):
    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self.encoder(x)
        val_loss = F.cross_entropy(y_hat, y)
        self.log("val_loss", val_loss)


        #Configure Prediction logic
class LitModel(L.LightningModule):
    def predict_step(self, batch, batch_idx):
        x, y = batch
        pred = self.encoder(x)
        return pred
    

    #Remove any .cuda() or .to(device) Calls
class LitModel(L.LightningModule):
    def training_step(self, batch, batch_idx):
        z = torch.randn(4, 5, device=self.device)
        ...
class LitModel(L.LightningModule):
    def __init__(self):
        super().__init__()
        self.register_buffer("running_mean", torch.zeros(num_features))
# 
#Additionally, you can run only the validation loop using validate() method.
model = LitModel()
trainer.validate(model)
#The test loop isn’t used within fit(), therefore, you would need to explicitly call test().
model = LitModel()
trainer.test(model)