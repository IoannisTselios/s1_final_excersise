import torch
from model import MyAwesomeModel  # Corrected import statement
import pytest

def test_model_output_shape():
    # Instantiate your model
    model = MyAwesomeModel()  # Corrected instantiation

    # Ensure the model is in evaluation mode
    model.eval()

    # Create a dummy input tensor with the expected input shape
    batch_size = 256
    dummy_input = torch.randn((batch_size, 1, 28, 28))

    # Forward pass
    output = model(dummy_input)

    # Assert the output shape
    expected_output_shape = (batch_size, 10)  # Assuming output size is (batch_size, 10)
    assert output.shape == expected_output_shape

def test_error_on_wrong_shape():
    # Instantiate your model
    model = MyAwesomeModel()

    # Test with a 3D tensor, should raise ValueError
    with pytest.raises(ValueError, match='Expected input to be a 4D tensor'):
        model(torch.randn(1, 2, 3))

    # Test with a tensor of incorrect shape [1, 32, 32], should raise ValueError
    with pytest.raises(ValueError, match='Expected each sample to have shape'):
        model(torch.randn(1, 1, 32, 32))

