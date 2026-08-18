import torch


def angles_to_unit_vectors(angles):
    pitch = angles[:, 0]
    yaw = angles[:, 1]

    x = -torch.cos(pitch) * torch.sin(yaw)
    y = -torch.sin(pitch)
    z = -torch.cos(pitch) * torch.cos(yaw)

    return torch.stack([x, y, z], dim=1)


def angular_error_degrees(predictions, labels):
    assert predictions.shape == labels.shape

    pred_vec = angles_to_unit_vectors(predictions)
    label_vec = angles_to_unit_vectors(labels)

    dot = (pred_vec * label_vec).sum(dim=1)
    dot = dot.clamp(-1.0, 1.0)

    return torch.rad2deg(torch.acos(dot))


if __name__ == "__main__":
    v = angles_to_unit_vectors(torch.tensor([[0.0, 0.0]]))
    assert torch.allclose(v[0], torch.tensor([0.0, 0.0, -1.0]))

    two = angles_to_unit_vectors(torch.tensor([[0.0, 0.5], [0.5, 0.0]]))
    assert not torch.allclose(two[0], two[1])
    assert torch.allclose((two ** 2).sum(dim=1), torch.ones(2))

    pred = torch.tensor([[0.13, 0.16]])
    truth = torch.tensor([[0.10, 0.20]])
    assert torch.allclose(angular_error_degrees(pred, truth)[0], torch.tensor(2.853), atol=0.01)
    assert torch.allclose(angular_error_degrees(truth, truth)[0], torch.tensor(0.0), atol=1e-5)

    print("all checks passed")