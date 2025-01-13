import torch
import torch.nn as nn
import unikan.basis as basis
from unikan import SKAN_pure

def test_custom_initialization():
    # Define custom initialization function
    def custom_init(weight):
        nn.init.kaiming_uniform_(weight, a=5 ** 0.5)
    
    # Select device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Create network with custom initialization
    net = SKAN_pure(
        layer_sizes=[2, 3, 1],
        basis_function=basis.lshifted_softplus,
        device=device,
        init_method=custom_init
    )
    
    # Test if weights are initialized as expected
    for layer in net.layers:
        for node in layer.nodes:
            for weight in node.weight:
                # Calculate fan_in
                fan_in = weight.size(1)
                if node.use_bias:
                    fan_in -= 1  # Subtract 1 for bias
                
                # Calculate bounds for kaiming uniform
                bound = (5 ** 0.5) * (3/fan_in) ** 0.5
                
                # Check if weights are within expected bounds
                assert torch.all(weight <= bound) and torch.all(weight >= -bound), \
                    f"Weights are not within expected bounds: [-{bound}, {bound}]"
                
                # Check if weights follow uniform distribution
                if node.use_bias:
                    weights_no_bias = weight[:, :-1]
                else:
                    weights_no_bias = weight
                    
                mean = weights_no_bias.mean().item()
                
                # Allow for some statistical variation in mean
                assert -0.5 < mean < 0.5, \
                    f"Mean of weights ({mean}) deviates too much from 0"

                # Verify that weights are within the expected range
                max_val = weights_no_bias.max().item()
                min_val = weights_no_bias.min().item()
                assert min_val >= -bound and max_val <= bound, \
                    f"Weights range [{min_val}, {max_val}] exceeds bounds [-{bound}, {bound}]"

    print("All initialization tests passed successfully!")

if __name__ == "__main__":
    test_custom_initialization() 