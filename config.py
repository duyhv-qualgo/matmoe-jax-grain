import os
from pathlib import Path
from dataclasses import dataclass, field
from typing import List
from datetime import datetime
import jax.numpy as jnp
import flax.struct

current_dir = Path(__file__).resolve().parent

@dataclass
class Config:
    # --- Project Metadata ---
    signature: str = 'v7_matmoe_dynamic_k_prod'
    version: str = "7.2.6"
    description: str = "MatMoE v2 Scale Factor + Grad Accum + Dynamic Top-K + Stop Gradient KL + Encoder MSE"
    seed: int = 42

    # --- Directory Allocation ---
    artifact_root: Path = Path('artifacts')
    material_dir: Path = field(init=False)
    training_artifact_dir: Path = field(init=False)

    # --- Data Paths ---
    path_data: Path = field(init=False)
    tokenizer_path_: Path = field(init=False)

    # --- Tokenizer Configuration ---
    tokenizer_name: str = "t5-en-vi-tokenizer-20k"
    tokenizer_version: str = "v3"
    new_special_tokens: List[str] = field(default_factory=list)

    # --- Data Processing (Metadata) ---
    dataset_name: str = "ura-hcmut/PhoMT"
    train_max_length_input: int = 256
    train_max_length_output: int = 256
    max_length_inference: int = 256
    lang_pair: str = 'en<->vi'

    # --- Dataset Configuration (HF Arrow + Grain) ---
    dataset_version: str = '2_0'
    dataset_path: Path = field(init=False)
    total_examples: int = 5_955_998

    # =======================================================
    # 🌟 HARDWARE SCALE UP
    # =======================================================
    num_accelerator: int = 2  # 🚀 Upgraded to 8 GPUs
    batch_size: int = 2 ** 6
    eval_batch_size: int = 2 ** 10
    test_batch_size: int = 2 ** 10
    grad_accum_steps: int = 4

    total_training_steps: int = 1_000_000

    # --- Single LR Decay Plan ---
    lr_schedule_type: str = "cosine"
    learning_rate: float = 0.01
    warmup_steps: int = 500
    decay_init_value: float = 1e-6
    decay_steps: int = 80_000
    decay_end_value: float = 1e-5

    # --- Logging & Validation Intervals ---
    log_interval: int = 100
    eval_interval: int = 2000
    preview_interval: int = 2000
    test_interval: int = 2000
    ckpt_interval: int = 2000
    model_save_interval: int = 2000
    max_checkpoints_to_keep: int = 100

    sig_time: datetime = field(default_factory=datetime.now)

    # --- Architecture ---
    d_model: int = 512
    num_heads: int = 8
    num_layers: int = 8
    d_ff: int = 1024
    dropout_rate: float = 0.1
    vocab_size: int = 0

    num_experts: int = 4
    top_k: int = 2
    semantic_dim: int = 64

    # --- Matryoshka Elastic MLP ---
    elastic_mlp_dims: List[int] = field(default_factory=lambda: [1024, 512, 256, 128])
    elastic_mlp_probs: List[float] = field(default_factory=lambda: [0.25, 0.25, 0.25, 0.25])
    use_r_drop: bool = True
    stop_gradient_kl: bool = False
    
    # --- Dynamic Top-K ---
    elastic_top_ks: List[int] = field(default_factory=lambda: [2, 1, 0])
    elastic_top_k_probs: List[float] = field(default_factory=lambda: [0.35, 0.35, 0.3])
    k_warmup_steps: int = 1000
    k_transition_steps: int = 20000

    # --- Dynamic Output Paths ---
    artifact_path: Path = field(init=False)
    tensorboard_log_path: Path = field(init=False)
    final_save_path: Path = field(init=False)
    latest_msg_path: Path = field(init=False)
    checkpoint_path: Path = field(init=False)
    tokenizer_path: Path = field(init=False)
    tokenizer_path_padded: Path = field(init=False)

    preview_texts: List[str] = field(default_factory=lambda: [
        "<translate-en-vi> Hello, how are you today?",
        "<translate-vi-en> Cảm ơn bạn rất nhiều vì sự giúp đỡ.",
        "<translate-en-vi> The weather is very beautiful today, so we should go for a walk in the park.",
        "<translate-vi-en> Bữa tối hôm nay rất ngon, tôi đặc biệt thích món cá nướng kiểu truyền thống.",
        "<translate-en-vi> Artificial intelligence and machine learning are rapidly transforming the landscape of modern technology and global economics.",
        "<translate-vi-en> Các nhà nghiên cứu đang phát triển những mô hình ngôn ngữ lớn nhằm cải thiện khả năng giao tiếp mượt mà giữa con người và máy tính.",
        "<translate-en-vi> The quick brown fox jumps over the lazy dog.",
        "<translate-vi-en> Trăm nghe không bằng một thấy, trăm thấy không bằng một thử.",
        "<translate-en-vi> The convergence of quantum computing and sparse neural networks presents an unprecedented opportunity for cryptographic advancements.",
        "<translate-vi-en> Một trong những thách thức cốt lõi của học máy lượng tử là việc duy trì trạng thái chồng chập trước hiện tượng mất kết hợp.",
        "<translate-en-vi> In the event of a material breach of this Agreement, the injured party is legally entitled to seek compensatory damages and injunctive relief.",
        "<translate-vi-en> Căn cứ vào các điều khoản quy định trong hợp đồng, người lao động có quyền đơn phương chấm dứt hợp đồng lao động nếu không được trả lương đầy đủ.",
        "<translate-en-vi> Don't count your chickens before they hatch; it's a piece of cake once you get the hang of it.",
        "<translate-vi-en> Trăm nghe không bằng một thấy, ngựa non thì thường hay háu đá.",
        "<translate-en-vi> The crimson sun dipped below the rugged horizon, casting elongated shadows that danced upon the tranquil surface of the ancient lake.",
        "<translate-vi-en> Tiếng mưa rơi rả rích trên mái tôn mỏng manh gợi lại trong lòng anh những kỷ niệm xa xăm về một tuổi thơ nghèo khó nhưng bình yên.",
        "<translate-vi-en> Chào, tui muốn đặt một ly trà sữa trân châu size lớn, ít đường nha!",
        "<translate-en-vi> I forgot my lines during the play, and it was so embarrassing in front of everyone!",
        "<translate-en-vi> Quantum computing could revolutionize cryptography, making current encryption methods obsolete in the near future.",
        "<translate-en-vi> Google is a multinational technology company known primarily for its search engine, but it also offers a wide range of other products and services. These include cloud computing, online advertising, software, and hardware like the Android operating system and Google Pixel phones. Google is a subsidiary of Alphabet Inc., a holding company that oversees various other ventures according to Britannica and Wikipedia.",
        "<translate-en-vi> OpenAI is used to develop and deploy advanced AI models, particularly in areas like natural language processing, image generation, and more. These models are used for a variety of applications, including powering chatbots, automating tasks, generating creative content, and analyzing data.",
        '<translate-en-vi> The sun sets slowly behind the mountain, casting a warm glow over the valley.',
    ])

    def __post_init__(self):
        self.material_dir = self.artifact_root / 'materials'
        self.training_artifact_dir = self.artifact_root / 'training_artifacts'
        self.path_data = self.material_dir / 'data'
        self.tokenizer_path_ = self.material_dir / 'tokenizer'

        dataset_name = f'phomt_{self.dataset_version}_{self.train_max_length_input}_{self.train_max_length_output}'
        self.dataset_path = self.path_data / dataset_name

        self.tokenizer_path = self.tokenizer_path_ / f'{self.tokenizer_name}-{self.tokenizer_version}'

        # 🌟 FIXED: Hardcoded to `_padded_4` so 8xH200 doesn't break the loading path!
        self.tokenizer_path_padded = self.tokenizer_path_ / f'{self.tokenizer_name}-{self.tokenizer_version}_padded_8'

        timestamp = self.sig_time.strftime("%Y_%m_%d_%H_%M")
        folder_stamp = f"{self.signature}_v{self.version}_in{self.train_max_length_input}_out{self.train_max_length_output}"

        self.artifact_path = self.training_artifact_dir / folder_stamp
        self.tensorboard_log_path = self.training_artifact_dir / 'tensorboard_logs' / folder_stamp

        self.checkpoint_path = self.artifact_path / 'checkpoints'
        self.final_save_path = self.artifact_path / f't5_model_{self.total_training_steps}steps_{timestamp}'
        self.latest_msg_path = self.artifact_path / "moe_model_latest.msg"

        self._create_dirs()

    def _create_dirs(self):
        dirs_to_create = [
            self.path_data, self.tokenizer_path_, self.tensorboard_log_path,
            self.checkpoint_path, self.final_save_path.parent
        ]
        for path in dirs_to_create:
            if not path.exists():
                path.mkdir(parents=True, exist_ok=True)


@flax.struct.dataclass
class MoEModelConfig:
    vocab_size: int = -1
    pad_token_id: int = -1
    vi_en_token_id: int = -1
    d_model: int = 512
    num_heads: int = 8
    mlp_dim: int = 1024
    num_layers: int = 8
    num_experts: int = 4
    top_k: int = 2
    semantic_dim: int = 64
    dropout_rate: float = 0.1
    max_seq_len: int = 128
    dtype: any = jnp.bfloat16


config = Config(
    new_special_tokens=["<translate-en-vi>", "<translate-vi-en>", "<tone-teen>", "<json>", "<cls>", "<info_ext>", "\n"]
)