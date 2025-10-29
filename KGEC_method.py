from typing import Literal, Optional, Union
import torch
from torch import Tensor, nn
from abc import ABC, abstractmethod
from calutils import expit_probs_x, expit_probs_binary, convert_to_tensor
import torch.optim as optim

Only_Use_CPU = True


class PostProcessing(ABC):
	def __init__(self):
		super().__init__()
		self.trained = False
		name: str

	def set_model(self) -> None:
		raise NotImplementedError

	@abstractmethod
	def fit(self, logits: Tensor, labels: Tensor) -> None:
		raise NotImplementedError

	@abstractmethod
	def predict(
			self,
			x: Tensor,
	) -> Tensor:
		raise NotImplementedError


class UncalCalibrator(PostProcessing):
	def __init__(self):
		super().__init__()
		self.name = 'Uncalibrated'

	def fit(self, uncal_probs, y_true):
		calibrated_probs = expit_probs_x(uncal_probs, np_ndarray=False)
		pass

	def predict(self, uncal_probs):
		calibrated_probs = expit_probs_x(uncal_probs, np_ndarray=False)
		calibrated_probs = convert_to_tensor(calibrated_probs)
		return calibrated_probs


class KGEC(nn.Module):
	def __init__(
			self,
			num_bins: int = 10,
			bin_edges: Optional[Tensor] = None,
			lr: float = 0.01,
			init_temp: float = 1.0,
			device: Optional[Union[Literal["cpu", "cuda"], torch.device]] = None,
	):
		super().__init__()
		self.num_bins = num_bins
		self.device = "cpu" if Only_Use_CPU else (device or ("cuda" if torch.cuda.is_available() else "cpu"))
		self.lr = lr
		self.init_temp = init_temp
		self.flag_show_temperature = False
		self.name = "KGEC"

		self.edges = torch.linspace(0, 1, num_bins + 1, device=self.device) if bin_edges is None else torch.tensor(
			bin_edges, dtype=torch.float32, device=self.device)
		self.bin_params = nn.Parameter(torch.full((num_bins,), init_temp, device=self.device))
		self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)
		self.to(self.device)

	def get_name(self) -> str:
		return self.name

	def scaling(self, probabilities, temperature):
		output = probabilities * (1 / torch.clamp((temperature ** 2), min=0.01, max=100))
		return output

	def forward(self, probabilities, jump_index=0, if_predict=False):
		sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
		jump_probs = sorted_probs[:, jump_index]
		bin_indices = torch.bucketize(jump_probs.contiguous(), self.edges) - 1
		bin_indices = bin_indices.clamp(0, self.num_bins - 1)
		bin_values = self.bin_params[bin_indices]
		output = self.scaling(jump_probs, bin_values)
		output = torch.log(output)
		if if_predict is True:
			calibrated_probs = self.scaling(probabilities, bin_values.unsqueeze(1))
		else:
			calibrated_probs = None

		return output, calibrated_probs

	def sinkhorn_normalized(self, x, n_iters=10):
		for _ in range(n_iters):
			x = x / torch.sum(x, dim=1, keepdim=True)
			x = x / torch.sum(x, dim=0, keepdim=True)
		return x

	def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=1):
		x, y = nn.Softmax(dim=1)(x.unsqueeze(0)), nn.Softmax(dim=1)(y.unsqueeze(0))
		Wxy = torch.cdist(x, y, p=1)
		K = torch.exp(-Wxy / epsilon)
		P = self.sinkhorn_normalized(K, n_iters)
		return torch.sum(P * Wxy)

	def fit(self, logits: Tensor, labels: Tensor, jump_index) -> None:
		self.jump_index = jump_index
		logits = expit_probs_x(logits).to(self.device)
		labels = labels.to(self.device)
		labels = expit_probs_binary(logits, labels).float().to(self.device)

		batch_size = 32
		for start in range(0, logits.size(0), batch_size):
			end = min(start + batch_size, logits.size(0))
			batch_logits = logits[start:end]
			batch_labels = labels[start:end]

			def calib_eval():
				self.optimizer.zero_grad()
				output, _ = self.forward(batch_logits, self.jump_index)
				loss = self.sinkhorn_loss(output, batch_labels)
				loss.backward()
				return loss

			self.optimizer.step(calib_eval)

	@torch.no_grad()
	def predict(self, logits: Tensor) -> Tensor:
		if not self.flag_show_temperature:
			print('self.bin_params:', self.bin_params)
			self.flag_show_temperature = True

		probabilities = expit_probs_x(logits).to(self.device)
		_, calibrated_probabilities = self.forward(probabilities, self.jump_index, if_predict=True)
		return calibrated_probabilities