# SNN-MNIST-Exploration

---

## 🚀 Project Overview

This repository delves into the fascinating world of **Spiking Neural Networks (SNNs)** by implementing a simple SNN model for the classic **MNIST handwritten digit classification** task. Unlike traditional Artificial Neural Networks (ANNs), SNNs operate using discrete spike events, mimicking the biological brain, which promises superior energy efficiency and processing capabilities for certain applications, especially on neuromorphic hardware.

This project specifically focuses on:
* Understanding the core concepts of SNNs, including spike encoding and spike sequences.
* Building an SNN classifier using the user-friendly `snnTorch` library, which leverages PyTorch's robust framework.
* Visualizing the dynamic behavior of SNNs, such as neuron spiking activity and membrane potentials, to gain intuitive insights.
* Demonstrating the training process of SNNs for a well-known dataset.

---

## ✨ Features

* **MNIST Data Preparation:** Custom spike encoding (e.g., Rate Coding) for static image data.
* **Simple SNN Architecture:** Implementation of a multi-layer SNN using `snntorch.Leaky` neurons.
* **Training & Evaluation:** Backpropagation Through Time (BPTT) with surrogate gradients for SNN training.
* **Spike Dynamics Visualization:** Plotting spike occurrences and neuron activations over time.
* **M1 Mac Optimization:** Utilizes PyTorch's Metal Performance Shaders (MPS) for accelerated training on Apple Silicon.

---

## 🛠️ Technologies Used

* **Python 3.x**
* **PyTorch:** The underlying deep learning framework.
* **snnTorch:** A PyTorch-based library for SNNs.
* **Matplotlib:** For data and spike visualization.
* **Jupyter Notebook/Lab:** For interactive development and presentation.
* **Anaconda/Miniforge:** For environment management.

---

## 🚀 Getting Started

Follow these steps to set up the environment and run the project locally.

### Prerequisites

* Ensure you have `conda` (Miniforge recommended for M1 Mac) installed.

### Installation

1.  **Clone the repository:**
    ```bash
    git clone [https://github.com/YourGitHubUsername/SNN-MNIST-Exploration.git](https://github.com/YourGitHubUsername/SNN-MNIST-Exploration.git)
    cd SNN-MNIST-Exploration
    ```
2.  **Create and activate a Conda environment:**
    ```bash
    conda create -n snn_env python=3.10
    conda activate snn_env
    ```
3.  **Install the required packages:**
    ```bash
    pip install torch torchvision torchaudio --index-url [https://download.pytorch.org/whl/cpu](https://download.pytorch.org/whl/cpu) # For Apple Silicon MPS
    pip install snntorch matplotlib ipykernel notebook # Or jupyterlab
    ```

### Running the Notebook

1.  **Launch Jupyter Notebook/Lab:**
    ```bash
    jupyter notebook
    ```
2.  **Open and run the notebook:**
    Navigate to the `snn_mnist_classifier.ipynb` (or similar) file and execute the cells.

---

## 📊 Results & Visualizations

(After running the notebook, you can add screenshots or descriptions here)

* **Training Loss & Accuracy Plots:** Showcase how the model's performance evolves over epochs.
* **Example Spike Encoded Image:** Illustrate how a static MNIST digit is converted into a spike sequence.
* **Output Neuron Spike Activity:** Visualize the firing patterns of the output neurons for a given input, demonstrating the SNN's decision-making process.

---

## 💡 Future Work

* Experiment with different spike encoding schemes (e.g., Time-To-First-Spike).
* Explore more complex SNN architectures (e.g., SNN-CNN hybrids).
* Investigate different SNN learning rules (e.g., STDP).
* Evaluate energy consumption metrics for the SNN.
* Apply the model to event-based datasets (e.g., N-MNIST) for true neuromorphic data.

---

## 🤝 Contributing

Contributions are welcome! If you have suggestions or improvements, please open an issue or submit a pull request.

---

## 📄 License

This project is licensed under the MIT License - see the `LICENSE` file for details.

---

## 📞 Contact

If you have any questions, feel free to reach out to [Your Name] at [Your Email Address].