"""
Contains pytorch modules and models for the MIL attention learning

M. Togninalli, 09.2019
"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from torchvision.models import resnet50, resnet18
from .cbn_resnet import cbn_resnet50, cbn_resnet18
from IPython import embed

# ------------------------------------------
# Encoders (ResNet, generalized ConvNets)
# ------------------------------------------
def transform_to_list(val, length):
    if not isinstance(val, list):
        return length*[val]
    else:
        assert len(val)==length
        return val

def get_output_dim(input_size, kernels, poolings, strides):
    # Recursive function to compute size
    output_size = ((input_size-kernels[0]+1)-poolings[0])/strides[0]+1
    if len(kernels)==1:
        return output_size
    else:
        return get_output_dim(output_size, kernels[1:], poolings[1:], strides[1:])

class ConvNetEncoder(nn.Module):
    def __init__(self, layers=[15, 20, 50, 50], in_channels=5, kernels=[5,5,3,3], poolings=2, strides=2, 
                 dropout=0.3, L=512, input_image_shape=(128,128)):
        """
        Generic CNN, 
        inputs: layers (list) indicate the depth of each convolution
                kernels (list, integer) indicate the kernel size at every layer, 
                        if integer repeats at every layer
                poolings (list, integer) indicate the pooling size at every pooling layer, 
                        if integer repeats at every layer
                stride (list, integer) indicate the stride size at every pooling layer, 
                        if integer repeats at every layer
                dropout (float) indicates the dropout probability to apply,
                L_in (int) indicates the encoded representation dimensionality
                input_image_shape is used to compute the final size of the convoluted images
        """
        super().__init__()
        self.L = L

        kernels = transform_to_list(kernels, len(layers))
        poolings = transform_to_list(poolings, len(layers))
        strides = transform_to_list(strides, len(layers))

        # Get the final output dimension (we only deal with squared images):
        self.output_dim = int(get_output_dim(input_image_shape[0], kernels, poolings, strides))
        print(self.output_dim)
        if self.output_dim < 1:
            raise Warning(f'The output dimension after convolutions and poolings is {output_dim}, ' +
                            f'please change kernels ({kernels}), poolings ({poolings}) or stride ({strides}) sizes ')
        
        
        self.conv_layers = nn.ModuleList([
            nn.Conv2d(a, b, kernel_size=k) for a,b,k in zip(
                [in_channels]+layers, layers, kernels)
        ])

        self.pooling_layers = nn.ModuleList([
            nn.MaxPool2d(p, stride=s) for p,s in zip(
                poolings, strides)
        ])

        self.dropout = nn.Dropout2d(p=dropout)

        self.fcn = nn.Linear(int(layers[-1]*self.output_dim*self.output_dim), self.L)

    def forward(self, x):
        # The trick is to mix all samples and then split them into instances again
        batch_size, bag_size, in_channels, img_size_x, img_size_y = x.size()
        x = x.view(batch_size*bag_size, in_channels, img_size_x, img_size_y)
        for conv_layer, pooling_layer in zip(self.conv_layers, self.pooling_layers):
            x = F.relu(conv_layer(x))
            x = pooling_layer(self.dropout(x))
        x = torch.flatten(x, 1)
        x = F.relu(self.fcn(x))
        # Reconstruct into batches
        x = x.view(batch_size,bag_size,self.L)
        return x

class ResnetEncoder(nn.Module):
    def __init__(self, in_channels=5, L=512, pretrained=True, resnet_version='18', cbn=False):
        """
        Loads Resnet architectures from torchvision
        inputs: 
                L (int) indicates the encoded representation dimensionality
                pretrained (bool) indicates whether to load a pretrained architecture
                resnet_version (str), choice between 18 or 50
                cbn (bool), use conditional batch norm for temporal encoding 
        """
        super().__init__()
        self.L = L

        if resnet_version=='50':
            if cbn:
                model = cbn_resnet50(pretrained=pretrained)
            else:    
                model = resnet50(pretrained=pretrained)
        elif resnet_version=='18':
            if cbn:
                model = cbn_resnet18(pretrained=pretrained)
            else:
                model = resnet18(pretrained=pretrained)
        else:
            raise Warning(f"This resnet version ({resnet_version}) does not exist.")

        # Number of filters in the bottleneck layer
        num_ftrs = model.fc.in_features

        # convert all the layers to list and remove the last one
        features = list(model.fc.children())[:-1]

        ## Add the last layer based on the num of classes in our dataset
        n_class=1
        features.extend([nn.Linear(num_ftrs, self.L)])

        ## convert it into container and add it to our model class.
        model.fc = nn.Sequential(*features)

        # Change first layer
        out_channels = model.conv1.out_channels
        kernel_size = model.conv1.kernel_size[0]
        pad = model.conv1.padding[0]
        stride = model.conv1.stride[0]

        new_features = nn.Conv2d(in_channels, out_channels, kernel_size=7, padding=3, stride=2)
        model.conv1 = new_features
        self.model = model

    def forward(self, x):
        # The trick is to mix all samples and then split them into instances again
        batch_size, bag_size, in_channels, img_size_x, img_size_y = x.size()
        x = x.view(batch_size*bag_size, in_channels, img_size_x, img_size_y)
        # TODO: only pass images that are not zeros
        # max_len = x.size()[1]
        # mask = get_mask_from_lengths(lengths, max_len, x.device)
        # x[~mask] = self.model(x[~mask])
        # Reconstruct into batches
        x = self.model(x)
        x = x.view(batch_size,bag_size,self.L)
        return x

# Genotype encoder
class GenotypeEncoder(nn.Module):
    def __init__(self, type_enc='mlp', in_size=35000,
                 n_hidden=[1024, 512], activation='relu',
                 dropout=0.3, L=512):
        """
        Generic Genotype encoder (MLP or CNN)
        inputs: type_enc (str)  determine the type of encoder, either mlp or cnn
                in_size (int)   input size of the genotypes (i.e. number of SNPs) 
                dropout (float) indicates the dropout probability to apply,
                L (int)         indicates the encoded representation dimensionality
        """
        super().__init__()
        self.type = type_enc
        self.L = L
        self.dropout = dropout
        if activation not in ['relu', 'leakyrelu', 'elu']:
            raise Warning('Activation must be either relu, leakyrelu or elu')
        activations = {
                'leakyrelu': nn.LeakyReLU(),
                'elu': nn.ELU(), 'relu': nn.ReLU()}
        self.activation = activations[activation]

        # Start simple: only mlp, for CNN implementation check convnet above
        sizes = [in_size] + n_hidden + [L]
        self.encoder = nn.ModuleList([
            nn.Linear(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)
        ])
        
    def forward(self, x):
        # Simple forward pass
        for enc_layer in self.encoder[:-1]:
            x = self.activation(enc_layer(x))
        # Do not use activation fn on last layer
        x = self.encoder[-1](x)
        return x


# ------------------------------------------
# MIL Aggregator
# ------------------------------------------
def get_mask_from_lengths(lengths, max_length=None, device=None):
    if max_length is None:
        max_length = lengths.max()

    indices = torch.arange(max_length, device=device).expand(len(lengths), max_length)

    mask = indices >= lengths.unsqueeze(1)
    return mask

class Aggregator(nn.Module):
    def __init__(self, encod_dim=512, D=128, heads=1):
        """
        Generic Aggregator
        inputs: encod_dim (int) indicates the dimension of the encodings
                D (int) indicates the dimension of the attention embedding
                output_dim (int) indicates the number of attention heads
        """
        super().__init__()
        self.L = encod_dim
        self.D = D
        self.n_heads = heads

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.n_heads)
        )

    def forward(self, x, lengths):
        max_len = x.size()[1]
        mask = get_mask_from_lengths(lengths, max_len, x.device)

        pre_attention = self.attention(x)  # BxNxK
        pre_attention[mask] = float('-inf')
        A = torch.transpose(pre_attention, 1, 2)  # BxKxN
        # print(A.size())
        A = F.softmax(A, dim=2)  # softmax over N
        # print(A)

        M = torch.bmm(A, x)  # BxKxL

        return M


# ------------------------------------------
# Temporal encodings
# ------------------------------------------

# class TemporalEncoding:
#     def __init__(self, min_time, max_time, n_dim=10):
#         self.max_time = max_time
#         self.n_dim = n_dim
#         self.setup_positional_encoding(min_time, max_time, n_dim)

#     def setup_positional_encoding(self, min_timescale, max_timescale,
#                                   n_channels):
#         # This is a bit hacky, but works
#         self.min_timescale = min_timescale
#         self.max_timescale = max_timescale
#         self.n_channels = n_channels
#         self.num_timescales = self.n_channels // 2
#         self.log_timescale_increment = (
#             math.log(float(self.max_timescale) / float(self.min_timescale)) /
#             (float(self.num_timescales) - 1))
#         self.inv_timescales = torch.Tensor(self.min_timescale * np.exp(
#             np.arange(self.num_timescales) * -self.log_timescale_increment
#         ))

#     def __call__(self, positions):
#         scaled_time = positions[:, :, None] * \
#                 self.inv_timescales[None, None, :]
#         signal = torch.cat(
#                 [torch.sin(scaled_time), torch.cos(scaled_time)], dim=-1)
#         return signal

# class PositionalEncoding():
#     """Apply positional encoding to instances."""

#     def __init__(self, min_timescale, max_timescale, n_channels):
#         """PositionalEncoding.
#         Args:
#             min_timescale: minimal scale of values
#             max_timescale: maximal scale of values
#             n_channels: number of channels to use to encode position
#         """
#         self.min_timescale = min_timescale
#         self.max_timescale = max_timescale
#         self.n_channels = n_channels

#         self._num_timescales = self.n_channels
#         self._inv_timescales = self._compute_inv_timescales()

#     def _compute_inv_timescales(self):
#         log_timescale_increment = (
#             math.log(float(self.max_timescale) / float(self.min_timescale))
#             / (float(self._num_timescales) - 1)
#         )
#         inv_timescales = (
#             self.min_timescale
#             * np.exp(
#                 np.arange(self._num_timescales)
#                 * -log_timescale_increment
#             )
#         )
#         return inv_timescales

#     def __call__(self, instance):
#         """Apply positional encoding to instances."""
#         # Extract the first two columns as these are the persistence tuples
#         encoded_val = []
#         for i in range(2):
#             c = instance[:, i]
#             scaled_time = (
#                 c[:, np.newaxis] *
#                 self._inv_timescales[np.newaxis, :]
#             )
#             signal = np.concatenate(
#                 (np.sin(scaled_time[:, ::2]), np.cos(scaled_time[:, 1::2])),
#                 axis=1)
#             encoded_val.append(signal)
#         # Concatenate and attach one-hot encoding
#         signal = np.hstack((*encoded_val, instance[:, 2:]))

#         return signal

# ------------------------------------------
# Full MIL learning with possibility to 
# extend to other data sources
# ------------------------------------------

class DeepMultiMIL(nn.Module):
    def __init__(self, encoder_type='convnet', encoder_options=None,
                    D=128, heads=1, temporal_encoding=False, n_dates=None):
        """
        MIL deep learning architecture
        """
        super().__init__()
        self.temporal_encoding = temporal_encoding
        assert encoder_type in ['convnet', 'resnet']
        enc = ConvNetEncoder if encoder_type=='convnet' else ResnetEncoder
        if encoder_options is not None:
            self.encoder = enc(**encoder_options)
        else:
            self.encoder = enc()

        # Aggregator
        if self.temporal_encoding:
            if not n_dates:
                raise Warning('You need to provide the number of dates.')
            self.aggregator = Aggregator(encod_dim=self.encoder.L+n_dates, D=D, heads=heads)
        else:
            self.aggregator = Aggregator(encod_dim=self.encoder.L, D=D, heads=heads)

        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.aggregator.L*self.aggregator.n_heads, 1)
        )
    
    def forward(self, x, lengths, dates=None):
        # TODO: to save on computation, avoid encoding empty images.
        x = self.encoder(x)
        if self.temporal_encoding:
            x = torch.cat([x,dates], dim=2)
        x = self.aggregator(x, lengths)
        batch_size, n_heads, L = x.size()
        x = x.view(batch_size, L*n_heads)
        return self.regressor(x)

# -------------------------------------------
# Flexible MIL model with multiple data types
# -------------------------------------------

class DeepMultiFusionMIL(nn.Module):
    def __init__(self, multispectral=True, encoder_type_multispectral='resnet', 
                    thermal=False, encoder_type_thermal='convnet', thermal_image_size=(40,40),
                    mid_processing=False, encoded_dim=256,
                    D=128, heads=1, concat_or_aggregate='aggregate',
                    temporal_encoding=False, n_dates=None,
                    channel_encoding=False):
        """
        MIL deep learning architecture for multiple data types
        args:
            - mid_processing is to add intermediate processing to the representations
        """
        super().__init__()
        assert encoder_type_multispectral in ['convnet', 'resnet']
        assert encoder_type_thermal in ['convnet', 'resnet']
        self.multispectral = multispectral
        self.thermal = thermal
        self.concat = concat_or_aggregate == 'concat' # Set this to true to concatenate the aggregated representation of the channels
        self.temporal_encoding = temporal_encoding
        self.n_data_channels = 0 # Count the number of channels for the encoding
        self.channel_encoding = channel_encoding


        # 1. have multiple encoders for each data type        
        if multispectral:
            self.multispectral_encoder = ConvNetEncoder(L=encoded_dim) if encoder_type_multispectral=='convnet' else ResnetEncoder(L=encoded_dim)
            self.n_data_channels += 1
        else:
            self.multispectral_encoder = None

        if thermal:
            if encoder_type_thermal=='convnet':
                self.thermal_encoder = ConvNetEncoder(in_channels=1,
                                                        layers=[10, 20, 40],
                                                        kernels=[5,5,3],
                                                        L=encoded_dim,
                                                        input_image_shape=thermal_image_size) 
            else:
                self.thermal_encoder = ResnetEncoder(in_channels=1, L=encoded_dim)
            self.n_data_channels += 1
        else:
            self.thermal_encoder = None

        # 2. Have mid-processing for encoded data representation (initially blank)
        # TODO

        if self.concat:
            # TODO: consider deprecate the concat feature, it is not converging at all.
            # 3. Aggregator
            if self.temporal_encoding:
                if not n_dates:
                    raise Warning('You need to provide the number of dates.')
                self.multispectral_aggregator = Aggregator(encod_dim=encoded_dim+n_dates, D=D, heads=heads)
                self.thermal_aggregator = Aggregator(encod_dim=encoded_dim+n_dates, D=D, heads=heads)
            else:
                self.multispectral_aggregator = Aggregator(encod_dim=encoded_dim, D=D, heads=heads)
                self.thermal_aggregator = Aggregator(encod_dim=encoded_dim, D=D, heads=heads)
            # 4. Regressor
            self.regressor = nn.Sequential(
                nn.Linear(2*self.thermal_aggregator.L*self.thermal_aggregator.n_heads, 1)
            )
        else:
            # Aggregator
            aggregator_dim = encoded_dim
            if self.temporal_encoding:
                if not n_dates:
                    raise Warning('You need to provide the number of dates.')
                aggregator_dim += n_dates

            if self.channel_encoding and self.n_data_channels > 1:
                # Add a one-hot encoding to indicate where the element comes from. 
                self.pad_0 = nn.ConstantPad1d((0,1),0.)
                self.pad_1 = nn.ConstantPad1d((0,1),1.)
                aggregator_dim += self.n_data_channels
            
            self.aggregator = FusedAggregator(encod_dim=aggregator_dim, D=D, heads=heads)
            # 4. Regressor
            self.regressor = nn.Sequential(
                nn.Linear(self.aggregator.L*self.aggregator.n_heads, 1)
            )

    def _unroll_input(self, x):
        # The elements always
        dates_multispectral, dates_thermal = None, None
        if self.temporal_encoding:
            # get actual date embeddings
            x, dates_multispectral, dates_thermal = x[:-2], x[-2], x[-1]
        if self.multispectral and self.thermal:
            x_multispectral, lengths_multispectral, x_thermal, lengths_thermal = x
        else:
            if self.multispectral:
                x_multispectral, lengths_multispectral = x
                x_thermal, lengths_thermal = None, None
            elif self.thermal:
                x_multispectral, lengths_multispectral = None, None
                x_thermal, lengths_thermal = x
            else:
                raise Warning('There must be at least a multispectral or a thermal input')
        return x_multispectral, lengths_multispectral, x_thermal, \
            lengths_thermal, dates_multispectral, dates_thermal
    
    def forward(self, x):
        # First unroll the input
        x_multispectral, lengths_multispectral, x_thermal, lengths_thermal, \
            dates_multispectral, dates_thermal  = self._unroll_input(x)
        # TODO: to save on computation, avoid encoding empty images.
        if self.multispectral:
            x_multispectral = self.multispectral_encoder(x_multispectral)
        if self.thermal:
            x_thermal = self.thermal_encoder(x_thermal)

        # If concatenate: aggregate thermal and multispec separately then concatenate
        if self.concat:
            x_multispectral = self.multispectral_aggregator(x_multispectral, lengths_multispectral)
            x_thermal = self.thermal_aggregator(x_thermal, lengths_thermal)
            if self.temporal_encoding:
                x_multispectral = torch.cat([x_multispectral,dates_multispectral], dim=2)
                x_thermal = torch.cat([x_thermal,dates_thermal], dim=2)
            # Concatenate the aggregated representations
            x = torch.cat([x_multispectral, x_thermal], dim=1)
        else:
            # Attn-Aggregate all together
            # Combine the embeddings
            if self.multispectral and self.thermal:
                if self.temporal_encoding:
                    # TODO: fix bug properly
                    try:
                        x_multispectral = torch.cat([x_multispectral,dates_multispectral], dim=2)
                        x_thermal = torch.cat([x_thermal,dates_thermal], dim=2)
                    except RuntimeError:
                        embed()
                        print('Warning: uneven lengths.')
                        pass
                if self.channel_encoding and self.n_data_channels > 1:
                    # Add encoding for each channel
                    x_multispectral = self.pad_0(self.pad_1(x_multispectral))
                    x_thermal = self.pad_1(self.pad_0(x_thermal))
                x = torch.cat([x_multispectral, x_thermal], dim=1)
                lengths = [lengths_multispectral, lengths_thermal]
            elif self.multispectral:
                x = x_multispectral
                lengths = [lengths_multispectral]
            else:
                x = x_thermal
                lengths = [lengths_thermal]
            # Aggregate
            x = self.aggregator(x, lengths)

        batch_size, n_heads, L = x.size()
        x = x.view(batch_size, L*n_heads)
        return self.regressor(x)

def get_mask_from_multiple_lengths(lengths, device=None):
    # Lengths can be high dimensional (matrix of num_sourcesxbatch_size)
    masks = []
    for l in lengths:
        max_length = l.max()
        indices = torch.arange(max_length, device=device).expand(len(l), max_length)
        mask = indices >= l.unsqueeze(1)
        masks.append(mask)
    # Group per sample in batch
    mask = torch.cat(masks, dim=1)
    return mask

class FusedAggregator(nn.Module):
    def __init__(self, encod_dim=512, D=128, heads=1):
        """
        Aggregator for multiple datasources
        inputs: encod_dim (int) indicates the dimension of the encodings
                D (int) indicates the dimension of the attention embedding
                output_dim (int) indicates the number of attention heads
        NOTE: THIS SEEMS TO BE GENERAL ENOUGH FOR THE OLD MODEL TOO (ONLY MULTISPECTRAL IMAGES)
        """
        super().__init__()
        self.L = encod_dim
        self.D = D
        self.n_heads = heads

        self.attention = nn.Sequential(
            nn.Linear(self.L, self.D),
            nn.Tanh(),
            nn.Linear(self.D, self.n_heads)
        )

    def forward(self, x, lengths):
        mask = get_mask_from_multiple_lengths(lengths, x.device)

        pre_attention = self.attention(x)  # BxNxK
        # TODO: fix the bug (IndexError: The shape of the mask [8, 31] at index 1 does not match the shape of the indexed tensor [8, 32, 4] at index 1)
        try:
            pre_attention[mask] = float('-inf')
        except IndexError:
            print('No masking could be done!')
            pass
        A = torch.transpose(pre_attention, 1, 2)  # BxKxN
        # print(A.size())
        A = F.softmax(A, dim=2)  # softmax over N
        # print(A)

        M = torch.bmm(A, x)  # BxKxL

        return M

class DeepGenoFusionMIL(nn.Module):
    def __init__(self, multispectral=True, encoder_type_multispectral='resnet', 
                    thermal=False, encoder_type_thermal='resnet', thermal_image_size=(40,40),
                    genotype=False, genotype_size=38361,
                    mid_processing=False, encoded_dim=256,
                    D=128, heads=1, concat_or_aggregate='aggregate',
                    temporal_encoding=False, n_dates=None,
                    channel_encoding=False):
        """
        MIL deep learning architecture for multiple data types (thermal, temporal, genotype)
        """
        super().__init__()
        assert encoder_type_multispectral in ['convnet', 'resnet']
        assert encoder_type_thermal in ['convnet', 'resnet']
        self.multispectral = multispectral
        self.thermal = thermal
        self.genotype = genotype
        self.concat = concat_or_aggregate == 'concat' # DEPRECATED FOR THIS
        self.temporal_encoding = temporal_encoding
        self.n_data_channels = 0 # Count the number of channels for the encoding
        self.channel_encoding = channel_encoding


        # 1. have multiple encoders for each data type        
        if multispectral:
            self.multispectral_encoder = ConvNetEncoder(L=encoded_dim) if encoder_type_multispectral=='convnet' else ResnetEncoder(L=encoded_dim)
            self.n_data_channels += 1
        else:
            self.multispectral_encoder = None

        if thermal:
            if encoder_type_thermal=='convnet':
                self.thermal_encoder = ConvNetEncoder(in_channels=1,
                                                        layers=[10, 20, 40],
                                                        kernels=[5,5,3],
                                                        L=encoded_dim,
                                                        input_image_shape=thermal_image_size) 
            else:
                self.thermal_encoder = ResnetEncoder(in_channels=1, L=encoded_dim)
            self.n_data_channels += 1
        else:
            self.thermal_encoder = None
        
        if genotype:
            self.genotype_encoder = GenotypeEncoder(in_size=genotype_size, L=encoded_dim)
            self.n_data_channels += 1

        # 2. Have mid-processing for encoded data representation (initially blank)
        # TODO

        # Aggregator
        aggregator_dim = encoded_dim
        if self.temporal_encoding:
            if not n_dates:
                raise Warning('You need to provide the number of dates.')
            aggregator_dim += n_dates

        if self.channel_encoding and self.n_data_channels > 1:
            # Add a one-hot encoding to indicate where the element comes from. 
            self.pad_0 = nn.ConstantPad1d((0,1),0.)
            self.pad_1 = nn.ConstantPad1d((0,1),1.)
            aggregator_dim += self.n_data_channels
        
        self.aggregator = FusedAggregator(encod_dim=aggregator_dim, D=D, heads=heads)
        # 4. Regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.aggregator.L*self.aggregator.n_heads, 1)
        )

    def _unroll_input(self, x):
        # The elements always
        dates_multispectral, dates_thermal = None, None
        if self.temporal_encoding:
            # get actual date embeddings
            x, dates_multispectral, dates_thermal = x[:-2], x[-2], x[-1]
        if self.multispectral and self.thermal and self.genotype:
            x_multispectral, lengths_multispectral, x_thermal, lengths_thermal, x_genotype = x
        else:
            if self.multispectral and self.genotype:
                # TODO: UNTESTED!!! 
                x_multispectral, lengths_multispectral, x_thermal, lengths_thermal, x_genotype = x
                x_thermal, lengths_thermal = None, None
            elif self.thermal and self.genotype:
                x_multispectral, lengths_multispectral, x_genotype = None, None
                x_thermal, lengths_thermal = x
            else:
                raise Warning('There must be at least a multispectral or a thermal input')
        return x_multispectral, lengths_multispectral, x_thermal, \
            lengths_thermal, x_genotype, dates_multispectral, dates_thermal
    
    def forward(self, x):
        # First unroll the input
        x_multispectral, lengths_multispectral, x_thermal, lengths_thermal, \
            x_genotype, dates_multispectral, dates_thermal  = self._unroll_input(x)
        # TODO: to save on computation, avoid encoding empty images.
        if self.multispectral:
            x_multispectral = self.multispectral_encoder(x_multispectral)
        if self.thermal:
            x_thermal = self.thermal_encoder(x_thermal)
        if self.genotype:
            x_genotype = self.genotype_encoder(x_genotype)

        # If concatenate: aggregate thermal and multispec separately then concatenate
        if self.concat:
            # NOT FUNCTIONAL
            x_multispectral = self.multispectral_aggregator(x_multispectral, lengths_multispectral)
            x_thermal = self.thermal_aggregator(x_thermal, lengths_thermal)
            if self.temporal_encoding:
                x_multispectral = torch.cat([x_multispectral,dates_multispectral], dim=2)
                x_thermal = torch.cat([x_thermal,dates_thermal], dim=2)
            # Concatenate the aggregated representations
            x = torch.cat([x_multispectral, x_thermal], dim=1)
        else:
            # Attn-Aggregate all together
            # Combine the embeddings
            if self.multispectral and self.thermal:
                if self.temporal_encoding:
                    # TODO: fix bug properly
                    # try:
                    x_multispectral = torch.cat([x_multispectral,dates_multispectral], dim=2)
                    x_thermal = torch.cat([x_thermal,dates_thermal], dim=2)
                    x_genotype = torch.cat([x_genotype, torch.zeros(x_genotype.size()[0],16, device=x_genotype.device)], dim=1)
                    # except RuntimeError:
                    #     print('Warning: uneven lengths.')
                    #     pass
                if self.channel_encoding and self.n_data_channels > 1:
                    # Add encoding for each channel
                    x_multispectral = self.pad_0(self.pad_0(self.pad_1(x_multispectral)))
                    x_thermal = self.pad_0(self.pad_1(self.pad_0(x_thermal)))
                    x_genotype = self.pad_1(self.pad_0(self.pad_0(x_genotype)))
                # Must align x_genotype with format of bags batch_size x 1 x embedding_dim
                x_genotype = x_genotype.unsqueeze(1)
                x = torch.cat([x_multispectral, x_thermal, x_genotype], dim=1)
                # Add genotype length for masking
                lengths_genotype = torch.ones_like(lengths_thermal)
                lengths = [lengths_multispectral, lengths_thermal, lengths_genotype]
            elif self.multispectral: # DEPRECATED
                if self.channel_encoding and self.n_data_channels > 1:
                    # Add encoding for each channel
                    x_multispectral = self.pad_0(self.pad_1(x_multispectral))
                    x_genotype = self.pad_1(self.pad_0(x_genotype))
                x_genotype = x_genotype.unsqueeze(1)
                x = torch.cat([x_multispectral, x_genotype], dim=1)
                # Add genotype length for masking
                lengths_genotype = torch.ones_like(lengths_multispectral)
                lengths = [lengths_multispectral, lengths_genotype]
            else:# DEPRECATED
                x = x_thermal
                lengths = [lengths_thermal]
            # Aggregate
            x = self.aggregator(x, lengths)

        batch_size, n_heads, L = x.size()
        x = x.view(batch_size, L*n_heads)
        return self.regressor(x)

# -------------------------------------------
# Genotype DeepLearning baseline model
# -------------------------------------------

class DeepGeno(nn.Module):
    def __init__(self, genotype_size=38361, encoded_dim=256,
                    activate_representation=False):
        """
        MIL deep learning architecture for multiple data types (thermal, temporal, genotype)
        """
        super().__init__()
        self.activate_representation = activate_representation
        if self.activate_representation:
            self.activation = nn.ReLU()
        # Encoder
        self.genotype_encoder = GenotypeEncoder(in_size=genotype_size, L=encoded_dim)

        # Regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.genotype_encoder.L, 1)
        )
    
    def forward(self, x):
        # First unroll the input
        x = self.genotype_encoder(x)
        if self.activate_representation:
            x = self.activation(x)
        return self.regressor(x)

# -------------------------------------------
# Digital Elevation Model
# -------------------------------------------
class FullFusionMIL(nn.Module):
    def __init__(self, multispectral=True, encoder_type_multispectral='resnet', 
                    thermal=False, encoder_type_thermal='resnet', 
                    thermal_image_size=(40,40),
                    dem=False, encoder_type_dem='resnet',
                    genotype=False, genotype_size=38361,
                    encoded_dim=256, D=128, heads=1,
                    temporal_encoding=False, n_dates=None,
                    channel_encoding=False):
        """
        MIL deep learning architecture for multiple data types (thermal, temporal, genotype)
        """
        super().__init__()
        assert encoder_type_multispectral in ['convnet', 'resnet']
        assert encoder_type_thermal in ['convnet', 'resnet']
        assert encoder_type_dem in ['convnet', 'resnet']
        self.multispectral = multispectral
        self.thermal = thermal
        self.dem = dem
        self.genotype = genotype
        self.temporal_encoding = temporal_encoding
        self.n_data_channels = 0 # Count the number of channels for the encoding
        self.channel_encoding = channel_encoding


        # 1. have multiple encoders for each data type        
        if multispectral:
            self.multispectral_encoder = ConvNetEncoder(L=encoded_dim) if encoder_type_multispectral=='convnet' else ResnetEncoder(L=encoded_dim)
            self.n_data_channels += 1
        else:
            self.multispectral_encoder = None

        if thermal:
            if encoder_type_thermal=='convnet':
                self.thermal_encoder = ConvNetEncoder(in_channels=1,
                                                        layers=[10, 20, 40],
                                                        kernels=[5,5,3],
                                                        L=encoded_dim,
                                                        input_image_shape=thermal_image_size) 
            else:
                self.thermal_encoder = ResnetEncoder(in_channels=1, L=encoded_dim)
            self.n_data_channels += 1
        else:
            self.thermal_encoder = None
        
        if dem:
            if encoder_type_dem=='convnet':
                self.dem_encoder = ConvNetEncoder(in_channels=1,
                                                        layers=[10, 20, 40],
                                                        kernels=[5,5,3],
                                                        L=encoded_dim,
                                                        input_image_shape=thermal_image_size) 
            else:
                self.dem_encoder = ResnetEncoder(in_channels=1, L=encoded_dim)
            self.n_data_channels += 1
        else:
            self.dem_encoder = None
        
        if genotype:
            self.genotype_encoder = GenotypeEncoder(in_size=genotype_size, L=encoded_dim)
            self.n_data_channels += 1

        # Aggregator
        aggregator_dim = encoded_dim
        if self.temporal_encoding:
            if not n_dates:
                raise Warning('You need to provide the number of dates.')
            aggregator_dim += n_dates

        if self.channel_encoding and self.n_data_channels > 1:
            # Add a one-hot encoding to indicate where the element comes from. 
            self.pad_0 = nn.ConstantPad1d((0,1),0.)
            self.pad_1 = nn.ConstantPad1d((0,1),1.)
            aggregator_dim += self.n_data_channels
        
        self.aggregator = FusedAggregator(encod_dim=aggregator_dim, D=D, heads=heads)
        # 4. Regressor
        self.regressor = nn.Sequential(
            nn.Linear(self.aggregator.L*self.aggregator.n_heads, 1)
        )

    def _unroll_input(self, x):
        # The elements always
        dates_multispectral, dates_thermal, dates_dem = None, None, None
        if self.temporal_encoding:
            # get actual date embeddings
            x, dates_multispectral, dates_thermal, dates_dem = x[:-3], x[-3], x[-2], x[-1]
        # if self.multispectral and self.thermal and self.dem and self.genotype:
        x_multispectral, lengths_multispectral, x_thermal, lengths_thermal, \
                x_dem, lengths_dem, x_genotype = x
        # else:
        #     raise Warning("This portion of the code is not tested (with not all the channels), proceed carefully.")
        #     if self.multispectral and self.genotype:
        #         # TODO: UNTESTED!!! 
        #         x_multispectral, lengths_multispectral, x_genotype = x
        #         x_thermal, lengths_thermal = None, None
        #     elif self.thermal and self.genotype:
        #         x_multispectral, lengths_multispectral, x_genotype = None, None
        #         x_thermal, lengths_thermal = x
        #     else:
        #         raise Warning('There must be at least a multispectral or a thermal input')
        return x_multispectral, lengths_multispectral, x_thermal, lengths_thermal, \
                x_dem, lengths_dem, x_genotype, dates_multispectral, dates_thermal, dates_dem
    
    def forward(self, x):
        # First unroll the input
        x_multispectral, lengths_multispectral, x_thermal, lengths_thermal, x_dem, lengths_dem, \
                x_genotype, dates_multispectral, dates_thermal, dates_dem  = self._unroll_input(x)
        # TODO: to save on computation, avoid encoding empty images.
        if self.multispectral:
            x_multispectral = self.multispectral_encoder(x_multispectral)
            if self.temporal_encoding:
                x_multispectral = torch.cat([x_multispectral,dates_multispectral], dim=2)
        if self.thermal:
            x_thermal = self.thermal_encoder(x_thermal)
            if self.temporal_encoding:
                x_thermal = torch.cat([x_thermal,dates_thermal], dim=2)
        if self.dem:
            x_dem = self.dem_encoder(x_dem)
            if self.temporal_encoding:
                x_dem = torch.cat([x_dem,dates_dem], dim=2)
        if self.genotype:
            x_genotype = self.genotype_encoder(x_genotype)
            if self.temporal_encoding:
                x_genotype = torch.cat([x_genotype, torch.zeros(x_genotype.size()[0],16, device=x_genotype.device)], dim=1)    

        # If concatenate: aggregate thermal and multispec separately then concatenate
        # Attn-Aggregate all together
        # Combine the embeddings
        if self.multispectral and self.thermal and self.dem and self.genotype:
            if self.channel_encoding and self.n_data_channels > 1:
                # Add encoding for each channel
                x_multispectral = self.pad_0(self.pad_0(self.pad_0(self.pad_1(x_multispectral))))
                x_thermal = self.pad_0(self.pad_0(self.pad_1(self.pad_0(x_thermal))))
                x_dem = self.pad_0(self.pad_1(self.pad_0(self.pad_0(x_dem))))
                x_genotype = self.pad_1(self.pad_0(self.pad_0(self.pad_0(x_genotype))))
            # Must align x_genotype with format of bags batch_size x 1 x embedding_dim
            x_genotype = x_genotype.unsqueeze(1)
            x = torch.cat([x_multispectral, x_thermal, x_dem, x_genotype], dim=1)
            # Add genotype length for masking
            lengths_genotype = torch.ones_like(lengths_dem)
            lengths = [lengths_multispectral, lengths_thermal, lengths_dem, lengths_genotype]
        elif self.multispectral and self.dem and self.genotype and not self.thermal: # No thermal
            if self.channel_encoding and self.n_data_channels > 1:
                # Add encoding for each channel
                x_multispectral = self.pad_0(self.pad_0(self.pad_1(x_multispectral)))
                x_dem = self.pad_0(self.pad_1(self.pad_0(x_dem)))
                x_genotype = self.pad_1(self.pad_0(self.pad_0(x_genotype)))
            # Must align x_genotype with format of bags batch_size x 1 x embedding_dim
            x_genotype = x_genotype.unsqueeze(1)
            x = torch.cat([x_multispectral, x_dem, x_genotype], dim=1)
            # Add genotype length for masking
            lengths_genotype = torch.ones_like(lengths_dem)
            lengths = [lengths_multispectral, lengths_dem, lengths_genotype]
        elif self.multispectral and self.genotype and not self.thermal and not self.dem: # No thermal and no dem
            if self.channel_encoding and self.n_data_channels > 1:
                # Add encoding for each channel
                x_multispectral = self.pad_0(self.pad_1(x_multispectral))
                x_genotype = self.pad_1(self.pad_0(x_genotype))
            # Must align x_genotype with format of bags batch_size x 1 x embedding_dim
            x_genotype = x_genotype.unsqueeze(1)
            x = torch.cat([x_multispectral, x_genotype], dim=1)
            # Add genotype length for masking
            lengths_genotype = torch.ones_like(lengths_multispectral)
            lengths = [lengths_multispectral, lengths_genotype]
        elif self.multispectral and self.dem and not self.thermal and not self.genotype: # No thermal nor genotype
            if self.channel_encoding and self.n_data_channels > 1:
                # Add encoding for each channel
                x_multispectral = self.pad_0(self.pad_1(x_multispectral))
                x_dem = self.pad_1(self.pad_0(x_dem))
            x = torch.cat([x_multispectral, x_dem], dim=1)
            # Add genotype length for masking
            lengths_genotype = torch.ones_like(lengths_dem)
            lengths = [lengths_multispectral, lengths_dem]
        elif self.multispectral and not self.dem and not self.thermal and not self.genotype: # Only MS
            x = x_multispectral
            # Add genotype length for masking
            lengths = [lengths_multispectral]
        else:# DEPRECATED
            raise Warning("Incomplete code for less channels.")
        # Aggregate
        x = self.aggregator(x, lengths)

        batch_size, n_heads, L = x.size()
        x = x.view(batch_size, L*n_heads)
        return self.regressor(x)

