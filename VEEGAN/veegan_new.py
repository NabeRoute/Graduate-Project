import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import torch.nn.functional as F
import pandas as pd
import matplotlib.pyplot as plt 
import torch.nn.utils.spectral_norm as spectral_norm
import torch.autograd



gpu = False
device = torch.device("cuda:0" if gpu and torch.cuda.is_available() else "cpu")

data_df = pd.read_excel('./data/MPEA_parsed_dataset.xlsx')
# Transform the dataset to a numpy array
data_np = data_df.to_numpy()[:, :]

# Identify the element molar ratios of the alloys and perform normalization
comp_data = data_np[:, 14:46].astype(float)
comp_min = comp_data.min(axis=0)
comp_max = comp_data.max(axis=0)
minmax_comp = (comp_data - comp_min) / comp_max

# Concatenate it with the processing data to produce the feature array X
proc_data = data_np[:, 46:53].astype(float)
X = np.concatenate((minmax_comp, proc_data), axis=1)


class GANTrainSet(Dataset):
    def __init__(self):
        """Initialize the dataset by converting the data array to a PyTorch tensor."""
        self.features = torch.from_numpy(X).float()
        self.len = self.features.shape[0]

    def __getitem__(self, index):
        """Retrieve a specific data item from the dataset using its index."""
        return self.features[index]

    def __len__(self):
        """Return the total number of data items in the dataset."""
        return self.len
    

class Reconstructor(nn.Module):
    def __init__(self):
        super(Reconstructor, self).__init__()
        
        self.model = nn.Sequential(
            nn.Linear(39, 10),  # 39は生成データの次元、10は潜在空間の次元
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
            nn.Linear(10, 10),
            nn.ReLU(),
        )

    def forward(self, x):
        return self.model(x)



class Generator(nn.Module):
    """
    Defines the Generator network within the Generative Adversarial Network (GAN).
    """

    def __init__(self):
        """Initialize the Generator model with fully connected layers and activation functions."""
        super(Generator, self).__init__()

        # The model consists of three layers with ReLU activation functions.
        # It aims to generate sparse non-zero outputs, mimicking realistic data.
        self.model = nn.Sequential(
            nn.Linear(10, 39),
            nn.ReLU(),
            nn.Linear(39, 39),
            nn.ReLU(),
            nn.Linear(39, 39),
            nn.ReLU(),
        )

    def forward(self, noise):
        """Execute a forward pass through the Generator network using the input noise."""
        fake_formula = self.model(noise)
        return fake_formula


class Discriminator(nn.Module):
    """
    Defines the Discriminator network within the Generative Adversarial Network (GAN).
    """

    def __init__(self):
        """Initialize the Discriminator model with fully connected layers and activation functions."""
        super(Discriminator, self).__init__()

        # The model consists of three layers with LeakyReLU activation functions.
        self.model = nn.Sequential(
            #spectral_norm(nn.Linear(39, 39)),  # 変更: spectral_normを適用
            nn.Linear(39, 39),
            nn.LeakyReLU(),
            #spectral_norm(nn.Linear(39, 39)),  # 変更: spectral_normを適用
            nn.Linear(39, 39),
            nn.LeakyReLU(),
        )

        #self.real_fake_output = spectral_norm(nn.Linear(39, 1))   # 変更: spectral_normを適用
        self.real_fake_output = nn.Linear(39, 1)

        self.latent_output = nn.Linear(39, 10)

    def forward(self, x):
        """Execute a forward pass through the Discriminator network using the input data."""
        feature = self.model(x)
        real_fake = self.real_fake_output(feature)
        latent_estimation = self.latent_output(feature)
        return real_fake, latent_estimation


def compute_gradient_penalty(d_net, real_samples, fake_samples):
    """
    Calculates the gradient penalty loss for Wasserstein GAN with Gradient Penalty (WGAN-GP).

    Parameters:
        d_net (nn.Module): Discriminator neural network.
        real_samples (Tensor): Real data samples.
        fake_samples (Tensor): Generated (fake) data samples.

    Returns:
        gp (Tensor): Computed gradient penalty.
    """

    # Generate random weight term alpha for interpolation between real and fake samples.
    alpha = torch.Tensor(np.random.random((real_samples.size(0), 1)))

    # Perform the interpolation between real and fake samples.
    # The interpolated sample 'interpolates' is set to require gradient to enable backpropagation.
    interpolates = (alpha * real_samples + ((1 - alpha) * fake_samples)).requires_grad_(True)

    # Evaluate the interpolated samples using the Discriminator network.
    d_interpolates_real_fake, d_interpolates_latent = d_net(interpolates)

    grad_outputs = torch.ones_like(d_interpolates_real_fake), torch.zeros_like(d_interpolates_latent)


    # Generate a tensor 'fake' to be used as grad_outputs for gradient computation.
    # It is set to not require gradient to prevent it from affecting loss minimization.
    #fake = torch.Tensor(real_samples.shape[0], 1).fill_(1.0)
    #fake.requires_grad = False

    # Compute gradients of the Discriminator outputs with respect to the interpolated samples.
    gradients = torch.autograd.grad(
        outputs=(d_interpolates_real_fake, d_interpolates_latent),
        inputs=interpolates,
        grad_outputs=grad_outputs,
        create_graph=True,
        retain_graph=True,
        only_inputs=True,
    )[0]

    # Reshape gradients to 2D tensor for ease of computation.
    gradients = gradients.view(gradients.size(0), -1)

    # Compute the gradient penalty based on the L2 norm of the gradients.
    gp = ((gradients.norm(2, dim=1) - 1) ** 2).mean()

    return gp


losses_g = []
losses_d = []


if __name__ == "__main__":
    # Hyperparameter for gradient penalty
    lambda_gp =  0.001

    # Initialize Generator and Discriminator models
    generator = Generator()
    discriminator = Discriminator()

    # Transfer the models to the appropriate computation device
    generator.to(device)
    discriminator.to(device)

    # Define the optimizers for both models
    reconstructor = Reconstructor().to(device)
    optimizer_R = torch.optim.Adam(reconstructor.parameters(), lr=1e-3)
    #optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-4, betas=(0, 0.999))   #変更ポイント(1or2)
    optimizer_G = torch.optim.Adam(generator.parameters(), lr=1e-3)
    #optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-4, betas=(0, 0.999))#変更ポイント(1or2)G=2e-4,D=2e-4, betas lossfig_GP4
                                                                                         #G=1e-4,D=1e-4 lossfig_GP3
    optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=1e-3)


    # Prepare the data loader
    al_data_set = GANTrainSet()
    loader = DataLoader(dataset=al_data_set, batch_size=5, shuffle=True)

    # Epochs for training
    for epoch in range(7000):
        # Initialize losses
        #loss_d_real = 0
        #loss_d_fake = 0
        #total_d_loss = 0

        epoch_g_loss = 0  
        epoch_d_loss = 0  
        epoch_r_loss = 0
        num_batches = 0 

        epoch_loss_d_real = 0  
        epoch_loss_d_fake = 0  

        # Batch processing
        for i, alloy in enumerate(loader):
            num_batches += 1
            real_input = alloy.to(device)

            # Discriminator training loop
            for j in range(5):
                # Generate fake data
                g_noise = torch.tensor(np.random.randn(alloy.shape[0], 10)).float()
                fake_alloy = generator(g_noise).detach() 

                optimizer_D.zero_grad()
                
                real_decision, _ = discriminator(real_input)
                real_decision = real_decision.view(-1)  # タプルから実数判定の値を取り出し、適切な形状に変換

                fake_decision, latent_estimation = discriminator(fake_alloy)
                fake_decision = fake_decision.view(-1)  # タプルから実数判定の値を取り出し、適切な形状に変換

                latent_loss = F.mse_loss(latent_estimation, g_noise)

                # Compute Gradient Penalty
                gradient_penalty = compute_gradient_penalty(discriminator, real_input.data, fake_alloy)

                # Calculate Discriminator Loss and Update Discriminator
                d_loss = -torch.mean(real_decision) + torch.mean(fake_decision) + latent_loss + lambda_gp * gradient_penalty 
                d_loss.backward()
                optimizer_D.step()

                epoch_d_loss += d_loss.item()
                epoch_r_loss += latent_loss.item()  

            # Generate fake data for Generator training
            #g_noise = torch.tensor(np.random.randn(alloy.shape[0], 10)).float()
            g_noise = torch.randn(alloy.shape[0], 10).to(device)
            fake_alloy = generator(g_noise)
            #fake_input = fake_alloy

            # Generator Update
            optimizer_G.zero_grad()
            optimizer_R.zero_grad()  

            fake_decision, latent_estimation = discriminator(fake_alloy)
            fake_decision = fake_decision.view(-1)

            g_loss = -torch.mean(fake_decision)
            r_loss = F.mse_loss(reconstructor(fake_alloy), g_noise)
            total_g_loss = g_loss + r_loss  # Generatorの全体の損失
            total_g_loss.backward()

            optimizer_G.step()
            optimizer_R.step()

            epoch_g_loss += g_loss.item()
            epoch_r_loss += r_loss.item()
        
        losses_g.append(epoch_g_loss / num_batches)
        losses_d.append(epoch_d_loss / num_batches)

        if epoch % 50 == 0:
            plt.figure(figsize=(10,5))
            plt.title("Generator and Discriminator Loss During Training")
            plt.plot(losses_g,label="G")
            plt.plot(losses_d,label="D")
            plt.xlabel("iterations")
            plt.ylabel("Loss")
            plt.legend()
            plt.rcParams["font.size"] = 20  
            plt.savefig(f"VEEGAN/VEEGAN_lossfig_7000epocs/plot_epoch_{epoch}.svg", bbox_inches="tight")  # 'lossfig' フォルダにプロットを保存
            plt.savefig(f"VEEGAN/VEEGAN_lossfig_7000epocs/plot_epoch_{epoch}.png", bbox_inches="tight") 
            plt.close()
           
           

        # Periodic Reporting and Visualization
        if epoch % 50 == 0:
            g_noise = torch.tensor(np.random.randn(3, 10)).float()
            fake_alloy = generator(g_noise)
            print(fake_alloy)

        # Compute the balance of the Discriminator
        denominator = epoch_loss_d_real + epoch_loss_d_fake
        balance = epoch_loss_d_real / denominator if denominator > 0 else 0

        # Logging
        if epoch < 500 or epoch % 20 == 0:
            print(epoch, "Discriminator balance:", balance, "D_loss:", epoch_d_loss / num_batches)

    # Save the trained Generator model
    torch.save(generator.state_dict(), 'VEEGAN_experiment_7000epocs_adam.pt')