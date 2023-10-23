import torch
import torch.nn as nn
import torch.nn.functional as F

# Define the VAE model
class VAE(nn.Module):

    # Define the Encoder
    class Encoder_RNN(nn.Module):
        def __init__(
                self, 
                emb_dim:int, 
                mid_dim:int, 
                latent_dim:int, 
                RNN_type:str, 
                RNN_options:dict,
                activation_fn:nn.Module,
                **kwargs
            ):
            super(VAE.Encoder_RNN, self).__init__()
            del kwargs

            # Define useful variables
            rnn_layer = VAE.RNN_map[RNN_type]
            bidirectional = RNN_options['bidirectional'] if 'bidirectional' in RNN_options else False
            lstm_out_dim = RNN_options['hidden_size'] * 2 if bidirectional else RNN_options['hidden_size']

            # layers
            self.embedding = nn.Embedding(VAE.N_EMB, emb_dim)
            self.lstm = rnn_layer(emb_dim, **RNN_options, batch_first=True)
            self.fc_post = nn.Sequential(nn.Linear(lstm_out_dim, mid_dim), activation_fn()) if mid_dim else nn.Identity()
            self.fc_mean    = nn.Linear(mid_dim if mid_dim else lstm_out_dim, latent_dim)
            self.fc_log_var = nn.Linear(mid_dim if mid_dim else lstm_out_dim, latent_dim)

        def forward(self, x):
            x = self.embedding(x)
            x, _ = self.lstm(x)
            x = x[:, -1, :]  # Take the last time step output
            x = self.fc_post(x)
            z_mean = self.fc_mean(x)
            z_log_var = self.fc_log_var(x)
            return z_mean, z_log_var
        
    # Define the Encoder
    class Encoder_conv(nn.Module):
        def __init__(
                self,
                convs_dims:list,
                convs_T_dims:list,
                mid_dim:int,
                latent_dim:int,
                activation_fn:nn.Module,
                **kwargs
            ):
            super(VAE.Encoder_conv, self).__init__()
            del kwargs

            # layers
            self.convs = nn.Sequential()
            last_dim = VAE.N_EMB if convs_T_dims else 0
            for conv_dim in convs_dims:
                conv_dim = [VAE.SQ_LEN if dim == 'in' else dim for dim in conv_dim]
                self.convs.append(nn.Conv1d(*conv_dim))
                self.convs.append(activation_fn())
                last_dim -= conv_dim[-1] - 1

            self.convs_T = nn.Sequential()
            last_T_dim = VAE.SQ_LEN if convs_T_dims else 0
            for conv_dim in convs_T_dims:
                conv_dim = [VAE.N_EMB if dim == 'in' else dim for dim in conv_dim]
                self.convs_T.append(nn.Conv1d(*conv_dim))
                self.convs_T.append(activation_fn())
                last_T_dim -= conv_dim[-1] - 1

            dim_in_linear = last_dim * self.convs[-1].out_channels + last_T_dim * self.convs_T[-1].out_channels
            self.fc_post = nn.Sequential(nn.Linear(dim_in_linear, mid_dim), activation_fn()) if mid_dim else nn.Identity()
            self.fc_mean    = nn.Linear(mid_dim if mid_dim else dim_in_linear, latent_dim)
            self.fc_log_var = nn.Linear(mid_dim if mid_dim else dim_in_linear, latent_dim)

        def forward(self, x):
            x = VAE.one_hot_encode(x)
            x = self.convs(x)
            x = x.view(x.size(0), -1)
            x_T = self.convs_T(x.transpose(-2, -1))
            x_T = x_T.view(x_T.size(0), -1)

            if len(self.convs)*len(self.convs_T) > 0:
                x = torch.cat([x, x_T], dim=-1) 
            elif len(self.convs) == 0:
                x = x_T

            x = self.fc_post(x.view(x.size(0), -1))
            z_mean = self.fc_mean(x)
            z_log_var = self.fc_log_var(x)
            return z_mean, z_log_var                     

    # Define the Decoder
    class Decoder(nn.Module):
        def __init__(
                self, 
                latent_dim:int, 
                output_dim:int, 
                mid_dim:int,
                RNN_type:str,
                RNN_options:dict,
                activation_fn:nn.Module,
                **kwargs
            ):
            super(VAE.Decoder, self).__init__()
            del kwargs

            # Define useful variables
            self.output_dim = output_dim
            rnn_layer = VAE.RNN_map[RNN_type]
            bidirectional = RNN_options['bidirectional'] if 'bidirectional' in RNN_options else False
            lstm_out_dim = RNN_options['hidden_size'] * 2 if bidirectional else RNN_options['hidden_size']

            # layers
            self.fc_pre = nn.Sequential(nn.Linear(latent_dim, mid_dim), activation_fn(), nn.Linear(mid_dim, latent_dim)) if mid_dim else nn.Sequential(nn.Linear(latent_dim, latent_dim))
            self.lstm = rnn_layer(latent_dim, **RNN_options, batch_first=True)
            self.fc_post = nn.Sequential(nn.Linear(lstm_out_dim, mid_dim), activation_fn(), nn.Linear(mid_dim, VAE.N_EMB), nn.Softmax(dim=-1)) if mid_dim else \
                           nn.Sequential(nn.Linear(lstm_out_dim, VAE.N_EMB), nn.Softmax(dim=-1))

        def forward(self, x):
            batch_size = x.size(0)
            x = self.fc_pre(x)
            x = x.unsqueeze(1).repeat(1, self.output_dim, 1)
            x, _ = self.lstm(x)              
            x = x.contiguous().view(batch_size, -1, x.size(-1))
            x = self.fc_post(x)
            return x

    # Define the hyperparameters
    N_EMB = 18
    SQ_LEN = 12 
    ENCODER_map = {'RNN': Encoder_RNN, 'conv': Encoder_conv}
    RNN_map = {'LSTM': nn.LSTM, 'GRU': nn.GRU, 'RNN': nn.RNN}

    one_hot_encode = lambda tensor: torch.zeros(*tensor.shape, VAE.N_EMB).scatter_(2, tensor.unsqueeze(2), 1)
    one_hot_decode = lambda tensor: torch.argmax(tensor, dim=-1).int()

    # Define the constructor
    def __init__(
            self, 
            emb_dim:int, 
            latent_dim:int, 
            output_dim:int,  
            mid_dim:int = None,
            encoder_type:str = 'RNN',
            RNN_type:str = 'LSTM',
            RNN_options:dict = {
                'hidden_size': 128,
                'bidirectional': True,
                'num_layers': 2,
                'dropout': 0.05,
            },
            activation_fn:nn.Module = nn.ReLU,
            **kwargs
        ):
        
        # Initialize the super class
        super(VAE, self).__init__()

        # Define the hyperparameters
        self.hyparams = {
            'emb_dim': emb_dim, 
            'latent_dim': latent_dim, 
            'output_dim': output_dim, 
            'mid_dim': mid_dim,
            'encoder_type': encoder_type,
            'RNN_type': RNN_type,
            'RNN_options': RNN_options, 
            'activation_fn': activation_fn,
            'kwargs': kwargs,
        }

        # Encoder and Decoder
        self.encoder = VAE.ENCODER_map[encoder_type](**self.hyparams) 
        self.decoder = VAE.Decoder(**self.hyparams)


    @staticmethod
    def reparameterize(encoder_out, weight_eps=1e-2):
        z_mean, z_log_var = encoder_out
        epsilon = torch.randn_like(z_mean)
        return z_mean + torch.exp(0.5 * z_log_var) * epsilon * weight_eps
    
    @staticmethod
    def loss_function(vae_out, x, reduction:str='sum', kl_weight:float=1.0):        
        reconstructed, mu, logvar = vae_out
        reconstruction_loss = F.cross_entropy(reconstructed.view(-1, VAE.N_EMB), x.view(-1).long(), reduction=reduction)
        kl_divergence = -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp())  
        return reconstruction_loss + kl_divergence*kl_weight
    
    @staticmethod
    def process_output(vae_out):
        return VAE.one_hot_decode(vae_out)
    
    def forward(self, x):
        z_mean, z_log_var = self.encoder(x)
        z = self.reparameterize((z_mean, z_log_var))
        return self.decoder(z), z_mean, z_log_var
    
    def optimization_step(self, batch, optimizer, loss_fn_options:dict={}):
        x, y = batch
        optimizer.zero_grad()
        loss = VAE.loss_function(self(x), x, **loss_fn_options)
        loss.backward()
        optimizer.step()
        return loss.item()
    
    def validation(self, batch, loss_fn_options:dict={}):
        x, y = batch
        with torch.no_grad():
            vae_out = self(x)
            loss = VAE.loss_function(vae_out, x, **loss_fn_options)

            x_pred = VAE.process_output(self.decoder(vae_out[1]))
            acc = (x_pred == x).float().mean()
        return loss.item(), acc.item()
    
    def save(self, path:str):
        torch.save(self, path)

    
    
