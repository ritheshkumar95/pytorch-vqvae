import pytest

import numpy as np
import torch

from functions import vq, vq_st

def test_vq_shape():
    inputs = torch.rand((2, 3, 5, 7), dtype=torch.float32, requires_grad=True)
    codebook = torch.rand((11, 7), dtype=torch.float32, requires_grad=True)
    indices = vq(inputs, codebook)

    assert indices.size() == (2, 3, 5)
    assert not indices.requires_grad
    assert indices.dtype == torch.int64

def test_vq():
    inputs = torch.rand((2, 3, 5, 7), dtype=torch.float32, requires_grad=True)
    codebook = torch.rand((11, 7), dtype=torch.float32, requires_grad=True)
    indices = vq(inputs, codebook)

    differences = inputs.unsqueeze(3) - codebook
    distances = torch.norm(differences, p=2, dim=4)

    _, indices_torch = torch.min(distances, dim=3)

    assert np.allclose(indices.numpy(), indices_torch.numpy())

def test_vq_st_shape():
    inputs = torch.rand((2, 3, 5, 7), dtype=torch.float32, requires_grad=True)
    codebook = torch.rand((11, 7), dtype=torch.float32, requires_grad=True)
    codes, indices = vq_st(inputs, codebook)

    assert codes.size() == (2, 3, 5, 7)
    assert codes.requires_grad
    assert codes.dtype == torch.float32

    assert indices.size() == (2 * 3 * 5,)
    assert not indices.requires_grad
    assert indices.dtype == torch.int64

def test_vq_st_gradient1():
    inputs = torch.rand((2, 3, 5, 7), dtype=torch.float32, requires_grad=True)
    codebook = torch.rand((11, 7), dtype=torch.float32, requires_grad=True)
    codes, _ = vq_st(inputs, codebook)

    grad_output = torch.rand((2, 3, 5, 7))
    grad_inputs, = torch.autograd.grad(codes, inputs,
        grad_outputs=[grad_output])

    # Straight-through estimator
    assert grad_inputs.size() == (2, 3, 5, 7)
    assert np.allclose(grad_output.numpy(), grad_inputs.numpy())

def test_vq_st_gradient2():
    inputs = torch.rand((2, 3, 5, 7), dtype=torch.float32, requires_grad=True)
    codebook = torch.rand((11, 7), dtype=torch.float32, requires_grad=True)
    codes, _ = vq_st(inputs, codebook)

    indices = vq(inputs, codebook)
    codes_torch = torch.embedding(codebook, indices, padding_idx=-1,
        scale_grad_by_freq=False, sparse=False)

    grad_output = torch.rand((2, 3, 5, 7), dtype=torch.float32)
    grad_codebook, = torch.autograd.grad(codes, codebook,
        grad_outputs=[grad_output])
    grad_codebook_torch, = torch.autograd.grad(codes_torch, codebook,
        grad_outputs=[grad_output])

    # Gradient is the same as torch.embedding function
    assert grad_codebook.size() == (11, 7)
    assert np.allclose(grad_codebook.numpy(), grad_codebook_torch.numpy())
