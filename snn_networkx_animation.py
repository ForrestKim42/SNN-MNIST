import torch
import torch.nn as nn
import snntorch as snn
from snntorch import surrogate
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
import networkx as nx
from torchvision import datasets, transforms
from torch.utils.data import DataLoader
# 사용하지 않는 임포트 제거

# 디바이스 설정
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
print(f"Using device: {device}")

# SNN 모델 클래스 정의 (기존과 동일)
class SNNNet(nn.Module):
    def __init__(self, num_inputs, num_hidden, num_outputs, beta):
        super().__init__()
        self.num_inputs = num_inputs
        self.num_hidden = num_hidden
        self.num_outputs = num_outputs
        self.beta = beta

        self.fc1 = nn.Linear(num_inputs, num_hidden)
        self.lif1 = snn.Leaky(beta=self.beta, spike_grad=surrogate.fast_sigmoid())
        self.fc2 = nn.Linear(num_hidden, num_outputs)
        self.lif2 = snn.Leaky(beta=self.beta, spike_grad=surrogate.fast_sigmoid())

    def forward(self, x):
        mem1 = self.lif1.init_leaky()
        mem2 = self.lif2.init_leaky()

        cur_list = []
        spk_list = []
        spk_hidden_list = []
        mem_list = []
        mem_hidden_list = []

        for step in range(x.size(0)):
            cur1 = self.fc1(x[step].view(-1, self.num_inputs))
            spk1, mem1 = self.lif1(cur1, mem1)
            cur2 = self.fc2(spk1)
            spk2, mem2 = self.lif2(cur2, mem2)

            cur_list.append(cur2)
            spk_list.append(spk2)
            spk_hidden_list.append(spk1)
            mem_list.append(mem2.clone())
            mem_hidden_list.append(mem1.clone())

        return (torch.stack(cur_list, dim=0), 
                torch.stack(spk_list, dim=0), 
                torch.stack(spk_hidden_list, dim=0),
                torch.stack(mem_list, dim=0),
                torch.stack(mem_hidden_list, dim=0))

# 저장된 모델 불러오기
def load_trained_model():
    model_path = 'snn_mnist_model.pth'
    checkpoint = torch.load(model_path, map_location=device)
    
    hyperparams = checkpoint['hyperparameters']
    
    net = SNNNet(
        hyperparams['num_inputs'],
        hyperparams['num_hidden'], 
        hyperparams['num_outputs'],
        hyperparams['beta']
    ).to(device)
    
    net.load_state_dict(checkpoint['model_state_dict'])
    net.eval()
    
    print(f"Model loaded successfully!")
    print(f"Architecture: {hyperparams['num_inputs']} -> {hyperparams['num_hidden']} -> {hyperparams['num_outputs']}")
    
    return net, hyperparams

# MNIST 데이터 준비
def prepare_mnist_sample():
    transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize((0.0,), (1.0,))
    ])
    
    mnist_test = datasets.MNIST(root='./data', train=False, download=False, transform=transform)
    test_loader = DataLoader(mnist_test, batch_size=1, shuffle=True)
    
    sample_image, sample_label = next(iter(test_loader))
    
    return sample_image, sample_label

# NetworkX 기반 SNN 시각화 클래스
class SNNNetworkXAnimator:
    def __init__(self, net, sample_image, sample_label, num_steps=25):
        self.net = net
        self.sample_image = sample_image
        self.sample_label = sample_label
        self.num_steps = num_steps
        self.device = device
        
        # 네트워크 구조 정보
        self.num_inputs = 784
        self.num_hidden = 100
        self.num_outputs = 10
        
        # 뉴런 활동 데이터 추출
        self._extract_neuron_activities()
        
        # NetworkX 그래프 생성
        self._create_network_graph()
        
        # 애니메이션 설정
        self._setup_animation()
        
    def _extract_neuron_activities(self):
        """모든 뉴런의 활동 데이터 추출"""
        print("Extracting neuron activities...")
        
        # 스파이크 시퀀스 생성
        spike_data = self.sample_image.unsqueeze(0).repeat(self.num_steps, 1, 1, 1, 1).to(self.device)
        
        with torch.no_grad():
            _, spk_out, spk_hidden, mem_out, mem_hidden = self.net(spike_data)
        
        # CPU로 이동하여 저장
        self.input_data = spike_data.cpu().numpy()  # (num_steps, 1, 1, 28, 28)
        self.spk_hidden = spk_hidden.cpu().numpy()  # (num_steps, 1, num_hidden)
        self.spk_out = spk_out.cpu().numpy()        # (num_steps, 1, num_outputs)
        self.mem_hidden = mem_hidden.cpu().numpy()  # (num_steps, 1, num_hidden)
        self.mem_out = mem_out.cpu().numpy()        # (num_steps, 1, num_outputs)
        
        print(f"Input shape: {self.input_data.shape}")
        print(f"Hidden layer spikes shape: {self.spk_hidden.shape}")
        print(f"Output layer spikes shape: {self.spk_out.shape}")
        
    def _create_network_graph(self):
        """NetworkX 그래프 생성"""
        print("Creating network graph...")
        
        self.G = nx.Graph()
        
        # 노드 추가
        # 입력층 노드 (784개 - 28x28 격자로 배치)
        input_nodes = []
        for i in range(28):
            for j in range(28):
                node_id = f"input_{i}_{j}"
                input_nodes.append(node_id)
                self.G.add_node(node_id, layer='input', pos=(j-14, i-14), pixel_idx=i*28+j)
        
        # 은닉층 노드 (100개 - 원형 배치)
        hidden_nodes = []
        for i in range(self.num_hidden):
            node_id = f"hidden_{i}"
            hidden_nodes.append(node_id)
            angle = 2 * np.pi * i / self.num_hidden
            x = 25 * np.cos(angle)
            y = 25 * np.sin(angle)
            self.G.add_node(node_id, layer='hidden', pos=(x, y), neuron_idx=i)
        
        # 출력층 노드 (10개 - 수직 배치)
        output_nodes = []
        for i in range(self.num_outputs):
            node_id = f"output_{i}"
            output_nodes.append(node_id)
            self.G.add_node(node_id, layer='output', pos=(40, (i-4.5)*3), neuron_idx=i)
        
        # 엣지 추가 (실제 가중치가 큰 연결만 표시)
        # 입력층 -> 은닉층 (샘플링해서 일부만)
        fc1_weights = self.net.fc1.weight.detach().cpu().numpy()  # (100, 784)
        
        # 각 은닉 뉴런에 대해 가장 강한 연결 10개만 표시
        for h_idx in range(self.num_hidden):
            weights = np.abs(fc1_weights[h_idx, :])
            top_indices = np.argsort(weights)[-20:]  # 상위 20개
            
            for input_idx in top_indices:
                if weights[input_idx] > 0.1:  # 임계값 이상만
                    input_row = input_idx // 28
                    input_col = input_idx % 28
                    input_node = f"input_{input_row}_{input_col}"
                    hidden_node = f"hidden_{h_idx}"
                    self.G.add_edge(input_node, hidden_node, weight=weights[input_idx])
        
        # 은닉층 -> 출력층 (모든 연결 표시)
        fc2_weights = self.net.fc2.weight.detach().cpu().numpy()  # (10, 100)
        
        for o_idx in range(self.num_outputs):
            for h_idx in range(self.num_hidden):
                weight = np.abs(fc2_weights[o_idx, h_idx])
                if weight > 0.05:  # 임계값 이상만
                    hidden_node = f"hidden_{h_idx}"
                    output_node = f"output_{o_idx}"
                    self.G.add_edge(hidden_node, output_node, weight=weight)
        
        self.input_nodes = input_nodes
        self.hidden_nodes = hidden_nodes
        self.output_nodes = output_nodes
        
        print(f"Graph created: {len(self.G.nodes)} nodes, {len(self.G.edges)} edges")
        
    def _setup_animation(self):
        """애니메이션 설정"""
        self.fig, self.ax_network = plt.subplots(1, 1, figsize=(16, 12))
        
        # 네트워크 시각화용 축
        self.ax_network.set_xlim(-50, 50)
        self.ax_network.set_ylim(-30, 20)
        self.ax_network.set_aspect('equal')
        self.ax_network.set_title('SNN Network Activity')
        
        # 노드 위치 추출
        self.pos = nx.get_node_attributes(self.G, 'pos')
        
    def animate(self, frame):
        """애니메이션 프레임 업데이트"""
        time_step = frame % self.num_steps
        
        # 축 초기화
        self.ax_network.clear()
        
        # 네트워크 축 설정
        self.ax_network.set_xlim(-50, 50)
        self.ax_network.set_ylim(-30, 20)
        self.ax_network.set_aspect('equal')
        self.ax_network.set_title(f'SNN Network Activity - Timestep: {time_step + 1}/{self.num_steps}')
        
        # 현재 시간 스텝의 활동 데이터
        input_activity = self.input_data[time_step, 0, 0, :, :].flatten()  # (784,)
        hidden_spikes = self.spk_hidden[time_step, 0, :]  # (100,)
        output_spikes = self.spk_out[time_step, 0, :]     # (10,)
        hidden_mem = self.mem_hidden[time_step, 0, :]     # (100,)
        output_mem = self.mem_out[time_step, 0, :]        # (10,)
        
        # 노드 색상과 크기 설정
        node_colors = []
        node_sizes = []
        
        for node in self.G.nodes():
            layer = self.G.nodes[node]['layer']
            
            if layer == 'input':
                pixel_idx = self.G.nodes[node]['pixel_idx']
                intensity = input_activity[pixel_idx]
                # 입력 픽셀 강도에 따라 색상 설정
                color = plt.cm.gray(intensity)
                size = 20 + 80 * intensity
                
            elif layer == 'hidden':
                neuron_idx = self.G.nodes[node]['neuron_idx']
                spike = hidden_spikes[neuron_idx]
                mem_volt = hidden_mem[neuron_idx]
                
                if spike > 0.5:
                    color = 'red'  # 스파이크 발생
                    size = 150
                else:
                    # 멤브레인 전압에 따른 색상
                    normalized_mem = (mem_volt + 1) / 2  # -1~1 -> 0~1
                    color = plt.cm.Blues(normalized_mem)
                    size = 50 + 100 * normalized_mem
                    
            else:  # output
                neuron_idx = self.G.nodes[node]['neuron_idx']
                spike = output_spikes[neuron_idx]
                mem_volt = output_mem[neuron_idx]
                
                if spike > 0.5:
                    color = 'red'  # 스파이크 발생
                    size = 200
                else:
                    # 멤브레인 전압에 따른 색상
                    normalized_mem = (mem_volt + 1) / 2
                    color = plt.cm.Greens(normalized_mem)
                    size = 80 + 120 * normalized_mem
            
            node_colors.append(color)
            node_sizes.append(size)
        
        # 엣지 그리기 (활성화된 연결만)
        active_edges = []
        edge_colors = []
        edge_widths = []
        
        for edge in self.G.edges():
            source, target = edge
            source_layer = self.G.nodes[source]['layer']
            
            # 소스 노드가 활성화되었는지 확인
            source_active = False
            if source_layer == 'input':
                pixel_idx = self.G.nodes[source]['pixel_idx']
                source_active = input_activity[pixel_idx] > 0.3
            elif source_layer == 'hidden':
                neuron_idx = self.G.nodes[source]['neuron_idx']
                source_active = hidden_spikes[neuron_idx] > 0.5
            
            if source_active:
                active_edges.append(edge)
                weight = self.G.edges[edge]['weight']
                edge_colors.append('red')
                edge_widths.append(weight * 3)
        
        # 네트워크 그리기
        nx.draw_networkx_nodes(self.G, self.pos, ax=self.ax_network,
                              node_color=node_colors, node_size=node_sizes, alpha=0.8)
        
        if active_edges:
            nx.draw_networkx_edges(self.G, self.pos, ax=self.ax_network,
                                  edgelist=active_edges, edge_color=edge_colors, 
                                  width=edge_widths, alpha=0.3)
        
        # 레이어 라벨 추가
        self.ax_network.text(-30, 15, 'Input Layer\n(28x28 pixels)', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightblue"))
        self.ax_network.text(-5, -25, 'Hidden Layer\n(100 neurons)', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightgreen"))
        self.ax_network.text(35, -25, 'Output Layer\n(10 classes)', 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightcoral"))
        
        # 스파이크 통계 표시
        hidden_spike_count = np.sum(hidden_spikes > 0.5)
        output_spike_count = np.sum(output_spikes > 0.5)
        predicted_class = np.argmax(output_spikes)
        
        stats_text = f"Active Neurons:\nHidden: {hidden_spike_count}/100\nOutput: {output_spike_count}/10\nPredicted: {predicted_class}"
        self.ax_network.text(-45, -25, stats_text, 
                           bbox=dict(boxstyle="round,pad=0.3", facecolor="lightyellow"))
        
        # # 입력 이미지 표시
        # self.ax_input.imshow(self.sample_image.squeeze().numpy(), cmap='gray')
        # self.ax_input.set_title(f'Input Image (Label: {self.sample_label.item()})')
        # self.ax_input.axis('off')
        
        return []
    
    def create_animation(self, interval=500, save_gif=False):
        """애니메이션 생성 및 실행"""
        print("Creating NetworkX animation...")
        
        if save_gif:
            print("Creating GIF animation...")
            # GIF용으로 프레임 수를 제한 (25개 타임스텝 * 2 사이클 = 50 프레임)
            ani = animation.FuncAnimation(self.fig, self.animate, frames=50, 
                                        interval=interval, blit=False, repeat=False)
            print("Saving animation as GIF...")
            ani.save('snn_networkx_animation.gif', writer='pillow', fps=2)
            print("Animation saved as 'snn_networkx_animation.gif'")
        else:
            # 실시간 애니메이션용 (무한 반복)
            ani = animation.FuncAnimation(self.fig, self.animate, frames=1000, 
                                        interval=interval, blit=False, repeat=True)
        
        plt.tight_layout()
        plt.show()
        
        return ani

# 메인 실행 함수
def main():
    print("=== SNN NetworkX Visualization ===")
    
    # 1. 모델 불러오기
    net, hyperparams = load_trained_model()
    
    # 2. 샘플 데이터 준비
    sample_image, sample_label = prepare_mnist_sample()
    print(f"Selected sample - Label: {sample_label.item()}")
    
    # 3. NetworkX 애니메이터 생성
    animator = SNNNetworkXAnimator(net, sample_image, sample_label, 
                                 num_steps=hyperparams['num_steps'])
    
    # 4. 애니메이션 실행 및 GIF 저장
    ani = animator.create_animation(interval=800, save_gif=True)
    
    return ani

if __name__ == "__main__":
    animation_obj = main()