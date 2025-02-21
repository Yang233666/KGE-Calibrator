from typing import Literal
import torch
from torch import Tensor, nn
from .calutils import expit_probs_x, expit_probs_binary, normalize
import torch.optim as optim

Only_Use_CPU = True


class KGEC(nn.Module):
	def __init__(self, num_bins=10, bin_edges=None, lr=0.01, max_iter: int = 100,
	             min_clamp: float = 0.01, max_clamp: float = 100,
	             device: Literal["cpu", "cuda"] | torch.device | None = None, ):
		super().__init__()
		self.num_bins = num_bins

		if Only_Use_CPU:
			self.device = "cpu"
		else:
			self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

		self.lr = lr
		self.max_iter = max_iter

		self.flag_show_temperature = False
		self.min_clamp = min_clamp
		self.max_clamp = max_clamp
		self.name = 'KGEC'

		if bin_edges is None:
			self.edges = torch.linspace(0, 1, num_bins + 1, device=self.device)
		else:
			self.edges = torch.tensor(bin_edges, device=self.device)

		self.bin_params = nn.Parameter(torch.ones(num_bins, device=self.device))
		self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)
		self.criterion = nn.KLDivLoss()

	def get_name(self) -> str:
		return self.name

	def scaling(self, probabilities, temperature):
		output = probabilities * (1 / torch.clamp((temperature ** 2), min=self.min_clamp, max=self.max_clamp))
		return output

	def forward(self, probabilities, if_predict=False):
		"""
		Forward pass that sorts the probability lists, finds the largest probability,
		and assigns it to the correct bin. The output is calculated as the largest
		probability divided by the corresponding bin value.

		Args:
			probabilities (Tensor): Input tensor of shape (batch_size, num_classes)

		Returns:
			Tensor: Output tensor of shape (batch_size,)
		"""

		sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
		calibrated_probabilities = torch.zeros_like(probabilities)

		max_probs = sorted_probs[:, 0]
		max_probs = normalize(max_probs)
		bin_indices = torch.bucketize(max_probs.contiguous(), self.edges) - 1
		bin_indices = bin_indices.clamp(0, self.num_bins - 1)
		bin_values = self.bin_params[bin_indices]
		output = self.scaling(max_probs, bin_values)
		if if_predict:
			for idx in range(max_probs.shape[0]):
				calibrated_probabilities[idx] = self.scaling(probabilities[idx], bin_values[idx])
		return output, calibrated_probabilities

	def fit(self, logits: Tensor, labels: Tensor) -> None:
		"""
		Performs a single training step using provided data.

		Args:
			logits (Tensor): Input probabilities, shape (batch_size, num_classes)
			labels (Tensor): Ground truth labels, shape (batch_size,)
		"""

		logits = expit_probs_x(logits)
		binary_labels = expit_probs_binary(logits, labels)
		logits, binary_labels = logits.to(self.device), binary_labels.to(self.device).float()

		batch_size = 32
		num_samples = logits.size(0)
		num_batches = (num_samples + batch_size - 1) // batch_size
		for batch_idx in range(num_batches):
			start_idx = batch_idx * batch_size
			end_idx = min(start_idx + batch_size, num_samples)

			self.batch_logits = logits[start_idx:end_idx, :]
			self.batch_binary_labels = binary_labels[start_idx:end_idx]

			def calib_eval():
				self.optimizer.zero_grad()
				output, calibrated_probabilities = self.forward(self.batch_logits)

				output = normalize(output)
				self.batch_binary_labels = normalize(self.batch_binary_labels)

				loss = self.criterion(output, self.batch_binary_labels)
				loss.backward()
				return loss

			self.optimizer.step(calib_eval)

	@torch.no_grad()
	def predict(self, logits: Tensor) -> Tensor:
		"""
		Predict based on the largest probability in each list.

		Args:
			logits (Tensor): Input probabilities, shape (batch_size, num_classes)

		Returns:
			Tensor: Output tensor of shape (batch_size,)
		"""

		if not self.flag_show_temperature:
			self.flag_show_temperature = True

		probabilities = expit_probs_x(logits).to(self.device)
		part1, calibrated_probabilities = self.forward(probabilities, if_predict=True)
		return calibrated_probabilities


class KGEC_plus(nn.Module):
	def __init__(self, num_bins=10, bin_edges=None, lr=0.01, max_iter: int = 100,
	             min_clamp: float = 0.01, max_clamp: float = 100, p: int = 1,
	             device: Literal["cpu", "cuda"] | torch.device | None = None, ):
		super().__init__()
		self.num_bins = num_bins

		if Only_Use_CPU:
			self.device = "cpu"
		else:
			self.device = device if device else ("cuda" if torch.cuda.is_available() else "cpu")

		self.lr = lr
		self.max_iter = max_iter

		self.flag_show_temperature = False
		self.min_clamp = min_clamp
		self.max_clamp = max_clamp
		self.p = p
		self.name = 'KGEC_plus'

		# Define bin edges if not provided
		if bin_edges is None:
			self.edges = torch.linspace(0, 1, num_bins + 1, device=self.device)
		else:
			self.edges = torch.tensor(bin_edges, device=self.device)

		# Define trainable parameters for each bin
		self.bin_params = nn.Parameter(torch.ones(num_bins, device=self.device))
		self.optimizer = optim.AdamW(self.parameters(), lr=self.lr)  # best
		self.to(self.device)

	def get_name(self) -> str:
		return self.name

	def scaling(self, probabilities, temperature):
		output = probabilities * (1 / torch.clamp((temperature ** 2), min=self.min_clamp, max=self.max_clamp))
		return output

	def forward(self, probabilities, if_predict=False):
		"""
		Forward pass that sorts the probability lists, finds the largest probability,
		and assigns it to the correct bin. The output is calculated as the largest
		probability divided by the corresponding bin value.

		Args:
			probabilities (Tensor): Input tensor of shape (batch_size, num_classes)

		Returns:
			Tensor: Output tensor of shape (batch_size,)
		"""
		# Sort probabilities for each input
		sorted_probs, _ = torch.sort(probabilities, dim=1, descending=True)
		calibrated_probabilities = torch.zeros_like(probabilities)
		max_probs = sorted_probs[:, 0]

		bin_indices = torch.bucketize(max_probs.contiguous(), self.edges) - 1
		bin_indices = bin_indices.clamp(0, self.num_bins - 1)

		bin_values = self.bin_params[bin_indices]
		output = self.scaling(max_probs, bin_values)
		if if_predict:
			for idx in range(max_probs.shape[0]):
				calibrated_probabilities[idx] = self.scaling(probabilities[idx], bin_values[idx])
		return output, calibrated_probabilities

	def sinkhorn_normalized(self, x, n_iters=10):
		for _ in range(n_iters):
			x = x / torch.sum(x, dim=1, keepdim=True)
			x = x / torch.sum(x, dim=0, keepdim=True)
		return x

	def sinkhorn_loss(self, x, y, epsilon=0.1, n_iters=1):
		softmax = nn.Softmax(dim=1)
		x = softmax(x.unsqueeze(0))
		y = softmax(y.unsqueeze(0))
		Wxy = torch.cdist(x, y, p=self.p)
		P = self.sinkhorn_normalized(torch.exp(-Wxy / epsilon), n_iters)
		emd_loss = torch.sum(P * Wxy)
		return emd_loss

	def fit(self, logits: Tensor, labels: Tensor) -> None:
		"""
		Performs a single training step using provided data.

		Args:
			logits (Tensor): Input probabilities, shape (batch_size, num_classes)
			labels (Tensor): Ground truth labels, shape (batch_size,)
		"""
		logits = expit_probs_x(logits)

		logits = logits.to(self.device)
		labels = labels.to(self.device)

		binary_labels = expit_probs_binary(logits, labels).float().to(self.device)

		batch_size = 32
		num_samples = logits.size(0)
		num_batches = (num_samples + batch_size - 1) // batch_size
		for batch_idx in range(num_batches):
			start_idx = batch_idx * batch_size
			end_idx = min(start_idx + batch_size, num_samples)
			batch_logits = logits[start_idx:end_idx, :]
			batch_binary_labels = binary_labels[start_idx:end_idx]

			def calib_eval():
				self.optimizer.zero_grad()
				output, calibrated_probabilities = self.forward(batch_logits)
				output = torch.log(output)
				loss = self.sinkhorn_loss(output, batch_binary_labels)
				loss.backward()
				return loss

			self.optimizer.step(calib_eval)

	@torch.no_grad()
	def predict(self, logits: Tensor) -> Tensor:
		"""
		Predict based on the largest probability in each list.

		Args:
			logits (Tensor): Input probabilities, shape (batch_size, num_classes)

		Returns:
			Tensor: Output tensor of shape (batch_size,)
		"""

		if not self.flag_show_temperature:
			self.flag_show_temperature = True

		probabilities = expit_probs_x(logits).to(self.device)
		part1, calibrated_probabilities = self.forward(probabilities, if_predict=True)
		return calibrated_probabilities
