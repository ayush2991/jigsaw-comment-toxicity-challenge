import streamlit as st
import pandas as pd
import logging
import torch
from torch.utils.data import DataLoader, Dataset
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.model_selection import train_test_split
import re
from collections import Counter
import altair as alt
import os

@st.cache_resource
def create_vocabulary(texts, min_freq=2):
    """Create and return a cached vocabulary instance"""
    vocab = Vocabulary(min_freq=min_freq)
    vocab.build_vocab(texts)
    return vocab

@st.cache_data(ttl=3600)
def load_and_prepare_data(target_label):
    """Load and prepare data with caching"""
    try:
        # Load data
        train_data = pd.read_csv('data/train.csv')
        
        # Basic preprocessing
        train_data['comment_text'] = train_data['comment_text'].fillna('')
        
        # Split data
        train_df, val_df = train_test_split(
            train_data[['comment_text', target_label]], 
            test_size=0.2, 
            random_state=42
        )
        
        # Create vocabulary from training data only
        vocab = create_vocabulary(train_df['comment_text'].values)
        
        return train_df, val_df, vocab
        
    except Exception as e:
        st.error(f"Error loading data: {e}")
        return None, None, None

class Vocabulary:
    def __init__(self, min_freq=2):
        self.word2idx = {}
        self.idx2word = {}
        self.word_freq = Counter()
        self.min_freq = min_freq
        self.special_tokens = ['<pad>', '<unk>', '<bos>', '<eos>']
        
        # Add special tokens
        for token in self.special_tokens:
            self.word2idx[token] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = token

    def build_vocab(self, texts):
        # Tokenize and count words
        for text in texts:
            words = self.tokenize(text)
            self.word_freq.update(words)
        
        # Add words that meet minimum frequency
        for word, freq in self.word_freq.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.word2idx[word] = len(self.word2idx)
                self.idx2word[len(self.idx2word)] = word

    def tokenize(self, text):
        # Simple tokenization: lowercase and split on whitespace
        text = text.lower()
        # Remove special characters but keep basic punctuation
        text = re.sub(r'[^a-z0-9\s.,!?]', '', text)
        return text.split()

    def __len__(self):
        return len(self.word2idx)

    def __getitem__(self, word):
        return self.word2idx.get(word, self.word2idx['<unk>'])

    def get_word(self, idx):
        return self.idx2word.get(idx, '<unk>')

    def get(self, word, default=None):
        # Mimic dict.get for compatibility
        return self.word2idx.get(word, self.word2idx['<unk>'] if default is None else default)
    
class ToxicityClassifier(nn.Module):
    def __init__(self, vocab_size, embedding_dim, hidden_dim):
        super(ToxicityClassifier, self).__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, 1)  # Output single value for binary classification

    def forward(self, x):
        x = self.embedding(x)  # [batch, seq_len, embedding_dim]
        x = x.mean(dim=1)     # [batch, embedding_dim] (mean pooling)
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x.squeeze(-1)  # [batch] for BCEWithLogitsLoss

# Define Dataset class at the module level for better organization
class ToxicityDataset(Dataset):
    def __init__(self, comments, labels, vocab, max_len=100):
        self.comments = comments.values  # Convert pandas Series to numpy array
        self.labels = labels.values      # Convert pandas Series to numpy array
        self.vocab = vocab
        self.max_len = max_len

    def __len__(self):
        return len(self.comments)

    def __getitem__(self, idx):
        text = self.comments[idx]
        tokens = self.vocab.tokenize(text)
        # Truncate or pad to max_len
        tokens = tokens[:self.max_len] if len(tokens) > self.max_len else tokens + [self.vocab.special_tokens[0]] * (self.max_len - len(tokens)) # Pad with '<pad>' token string
        indices = [self.vocab[word] for word in tokens] # vocab[word] handles OOV with <unk>
        return torch.tensor(indices), torch.tensor(self.labels[idx], dtype=torch.float32)

@st.cache_resource(ttl=3600)
def load_model(target_label, device):
    try:
        model_path = os.path.join('assets', target_label + ".pth")
        logger = logging.getLogger(__name__)
        model = torch.load(model_path, map_location=device, weights_only=False)
        model.eval()
        logger.info(f"Successfully loaded model for {target_label} classification")
        return model
    except Exception as e:
        logger.error(f"Failed to load model: {e}")
        st.error(f"Could not load pretrained model: {e}. Please train the model first.")
        return None

def main():
    # Set up logging
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    logger.info("Starting the app")

    st.set_page_config(
        page_title="Jigsaw Comment Toxicity Challenge",
        page_icon="🧩",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    st.title("Welcome to Jigsaw Comment Toxicity Challenge")
    tab_train, tab_predict = st.tabs(["Train", "Predict"])

    with tab_train:
        with st.sidebar:
            target_label = st.selectbox("Target Label", options=['toxic', 'severe_toxic', 'obscene', 'identity_hate', 'insult', 'threat'], index=0, key="target_label")
            st.header("Actions")
            train_button = st.button("Train Model", key="train_button")
            if st.button("Reset App State", key="reset_button"):
                st.experimental_rerun()
            st.subheader("Model Parameters")
            use_gpu = st.checkbox("Use GPU (if available)", value=True, key="use_gpu")
            quick_training_mode = st.checkbox("Dummy training mode", value=False, key="quick_training")
            save_model_to_disk = st.checkbox("Save Model to Disk", value=False, key="save_model")
            if use_gpu:
                if not torch.cuda.is_available():
                    device = torch.device("cpu")
                    st.warning("GPU is not available. Using CPU instead.")
                elif torch.cuda.is_available():
                    device = torch.device("cuda")
                    st.success("Using GPU for training.")
            else:
                device = torch.device("cpu")
                st.info("Using CPU for training.")
            num_epochs_val = st.slider("Number of Epochs", 1, 5, 1, key="num_epochs")
            learning_rate_val = st.select_slider("Learning Rate", options=[0.0001, 0.0005, 0.001, 0.005, 0.01, 0.1], value=0.01, key="lr")
            embedding_dim_val = st.select_slider("Embedding Dimension", options=[16, 32, 64, 128], value=32, key="embed_dim")
            hidden_dim_val = st.select_slider("Hidden Dimension", options=[16, 32, 64, 128], value=32, key="hidden_dim")
            max_seq_len_val = st.select_slider("Max Sequence Length", options=[50, 100, 150, 200], value=100, key="max_len")
            batch_size_val = st.select_slider("Batch Size", options=[16, 32, 64, 128], value=64, key="batch_size")

        # Add caching to load data
        @st.cache_data(ttl=3600)
        def load_data():
            try:
                train_data = pd.read_csv("data/train.csv")
                test_X = pd.read_csv("data/test.csv")
                test_Y = pd.read_csv("data/test_labels.csv")
                test_data = pd.merge(test_X, test_Y, on="id")
                return train_data, test_data
            except FileNotFoundError as e:
                logger.error(f"File not found: {e}")
                st.error("Data files not found. Please check the file paths.")
                return None, None
            except Exception as e:
                logger.error(f"An error occurred while loading data: {e}")
                st.error("An error occurred while loading data.")
                return None, None
            return train_data, test_data

        with st.spinner("Loading data..."):
            train_data, test_data = load_data()
            if train_data is None or test_data is None:
                st.stop()

        class_counts = train_data[target_label].value_counts().sort_index()
        class_df = pd.DataFrame({'class': ['non-'+target_label, target_label], 'count': [class_counts.get(0, 0), class_counts.get(1, 0)]})
        with st.expander("View Class Distribution Chart", expanded=False):
            st.bar_chart(class_df.set_index('class'), use_container_width=True)

        # Preview a sample of the data
        st.write("Sample Data")
        st.dataframe(train_data.sample(100, random_state=42), use_container_width=True, hide_index=True, height=200)

        with st.spinner("Processing data..."):            
            train_data, val_data = train_test_split(train_data, test_size=0.2, stratify=train_data[target_label], random_state=42)
            logger.info("Data split into training and validation sets")

            # Use only the target label for binary classification
            label_col = target_label
            if label_col not in train_data.columns:
                st.error(f"Label column '{label_col}' not found in training data.")
                return
            if label_col not in val_data.columns:
                st.error(f"Label column '{label_col}' not found in validation data.")
                return
            X_train = train_data['comment_text']
            y_train = train_data[label_col]
            X_val = val_data['comment_text']
            y_val = val_data[label_col]

            # Create and build vocabulary
            vocab = Vocabulary(min_freq=2)
            vocab.build_vocab(X_train)
            
            # Log vocabulary size
            logger.info(f"Vocabulary size: {len(vocab)}")

            # Create datasets and dataloaders
            train_dataset = ToxicityDataset(X_train, y_train, vocab, max_len=max_seq_len_val)
            val_dataset = ToxicityDataset(X_val, y_val, vocab, max_len=max_seq_len_val)
            train_loader = DataLoader(train_dataset, batch_size=batch_size_val, shuffle=True)
            val_loader = DataLoader(val_dataset, batch_size=batch_size_val, shuffle=False)

            model = ToxicityClassifier(len(vocab), embedding_dim=embedding_dim_val, hidden_dim=hidden_dim_val).to(device)

            logger.info("Data loaded into DataLoader")
            st.success("Data loaded successfully")

        # Train the model
        if train_button:
            with st.spinner("Training model..."):
                optimizer = optim.Adam(model.parameters(), lr=learning_rate_val)

                # Compute class weights for target label
                num_pos = (y_train == 1).sum()
                num_neg = (y_train == 0).sum()
                weight = num_neg / num_pos if num_pos > 0 else 1.0
                pos_weight = torch.tensor([weight], dtype=torch.float32, device=device)
                logger.info(f"Positive class weight: {pos_weight.item()}")

                # Use pos_weight in BCEWithLogitsLoss
                criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)

                # Placeholders for live updates
                status_placeholder = st.empty()
                chart_placeholder = st.empty()
                loss_history_df = pd.DataFrame(columns=['global_step', 'train_loss', 'val_loss'])
                global_step = 0

                for epoch in range(num_epochs_val):
                    model.train()
                    total_train_loss = 0
                    batch_count = 0
                    
                    for batch_idx, batch in enumerate(train_loader):
                        comments, labels = batch
                        comments, labels = comments.to(device), labels.to(device).float()
                        optimizer.zero_grad()
                        outputs = model(comments)
                        loss = criterion(outputs, labels)
                        loss.backward()
                        optimizer.step()
                        
                        total_train_loss += loss.item()
                        batch_count += 1
                        global_step +=1

                        # Update plot every 10th batch
                        if (batch_idx + 1) % 10 == 0 or (batch_idx + 1) == len(train_loader):
                            avg_epoch_train_loss = total_train_loss / batch_count
                            model.eval()
                            val_loss = 0
                            with torch.no_grad():
                                for val_batch in val_loader:
                                    val_comments, val_labels = val_batch
                                    val_comments = val_comments.to(device)
                                    val_labels = val_labels.to(device).float()
                                    val_outputs = model(val_comments)
                                    val_loss += criterion(val_outputs, val_labels).item()
                            val_loss = val_loss / len(val_loader) if len(val_loader) > 0 else 0
                            model.train() # Set back to train mode

                            # Update loss history using add_rows
                            new_data = pd.DataFrame({
                                'global_step': [global_step],
                                'train_loss': [avg_epoch_train_loss],
                                'val_loss': [val_loss]
                            })
                            loss_history_df = pd.concat([loss_history_df, new_data], ignore_index=True)
                            
                            # Update the line chart
                            chart_placeholder.line_chart(loss_history_df.set_index('global_step'))
                            
                            # Log progress
                            status_message = (f"Epoch {epoch+1}/{num_epochs_val}, Batch {batch_idx+1}/{len(train_loader)} | "
                                            f"Train Loss: {avg_epoch_train_loss:.4f} | Val Loss: {val_loss:.4f}")
                            status_placeholder.text(status_message)
                            logger.info(status_message)

                        if quick_training_mode and batch_idx >= 50:
                            break

                st.success("Training completed!")
                if save_model_to_disk:
                    os.makedirs('assets', exist_ok=True)
                    model_path = os.path.join('assets', target_label + ".pth")
                    torch.save(model, model_path)
                    st.success("Model saved to disk.")

                # Show prediction probabilities on 5 target_label samples and 5 non-target_label samples
                st.subheader("Sample Predictions")
                sample_data = pd.concat([val_data[val_data[target_label] == 1].sample(5, random_state=42),
                                        val_data[val_data[target_label] == 0].sample(5, random_state=42)])
                sample_data['predicted'] = 0.0
                for i, row in sample_data.iterrows():
                    comment = row['comment_text']
                    tokens = vocab.tokenize(comment)
                    if len(tokens) > max_seq_len_val:
                        tokens = tokens[:max_seq_len_val]
                    else:
                        tokens = tokens + [vocab.special_tokens[0]] * (max_seq_len_val - len(tokens))
                    indices = [vocab[word] for word in tokens]
                    input_tensor = torch.tensor(indices, device=device).unsqueeze(0)
                    with torch.no_grad():
                        outputs = model(input_tensor)
                        sample_data.at[i, 'predicted'] = torch.sigmoid(outputs).item()
                    # Show dataframe
                st.dataframe(sample_data[['comment_text', target_label, 'predicted']].reset_index(drop=True), use_container_width=True, hide_index=True)

                    
                

    with tab_predict:
        st.subheader("Predict Toxicity for Your Comment")
        target_label = st.selectbox("Target Label", options=['toxic', 'severe_toxic', 'obscene', 'identity_hate', 'insult', 'threat'], index=0, key="target_label_predict")
        
        # Check if model file exists before trying to load it
        model_path = os.path.join('assets', target_label + ".pth")
        logger.info(f"Checking for model at {model_path}")
        if not os.path.exists(model_path):
            logger.warning(f"No model file found at {model_path}")
            st.error(f"No trained model found for '{target_label}'. Please: \n1. Go to the Train tab\n2. Select '{target_label}' from the Target Label dropdown\n3. Check 'Save Model to Disk'\n4. Click 'Train Model'")
            st.stop()
            
        with st.spinner("Loading model..."):
            model = load_model(target_label, device)
            if model is None:
                logger.error("Model loading returned None")
                st.stop()
            logger.info(f"Model loaded successfully, device: {next(model.parameters()).device}")
        user_comment = st.text_area("Enter your comment here:", height=100, key="predict_text")
        predict_button = st.button("Predict Toxicity", key="predict_button")
       
        if predict_button:
            # Load pretrained model and vocab
            if user_comment:
                with st.spinner("Analyzing comment..."):
                    tokens = vocab.tokenize(user_comment)
                    if len(tokens) > max_seq_len_val:
                        tokens = tokens[:max_seq_len_val]
                    else:
                        tokens = tokens + [vocab.special_tokens[0]] * (max_seq_len_val - len(tokens))
                    indices = [vocab[word] for word in tokens]
                    input_tensor = torch.tensor(indices, device=device).unsqueeze(0)
                    outputs = model(input_tensor)
                    user_comment_predicted = torch.sigmoid(outputs).item()
                    st.write(f"Predicted Toxicity for Your Comment: {user_comment_predicted:.4f}")
            else:
                st.warning("Please enter a comment to predict its toxicity.")

         # Show prediction probabilities on 5 target_label samples and 5 non-target_label samples
        st.subheader("Sample Predictions")
        sample_data = pd.concat([val_data[val_data[target_label] == 1].sample(5, random_state=42),
                                val_data[val_data[target_label] == 0].sample(5, random_state=42)])
        sample_data['predicted'] = 0.0
        for i, row in sample_data.iterrows():
            comment = row['comment_text']
            tokens = vocab.tokenize(comment)
            if len(tokens) > max_seq_len_val:
                tokens = tokens[:max_seq_len_val]
            else:
                tokens = tokens + [vocab.special_tokens[0]] * (max_seq_len_val - len(tokens))
            indices = [vocab[word] for word in tokens]
            input_tensor = torch.tensor(indices, device=device).unsqueeze(0)
            with torch.no_grad():
                outputs = model(input_tensor)
                sample_data.at[i, 'predicted'] = torch.sigmoid(outputs).item()
            # Show dataframe
        st.dataframe(sample_data[['comment_text', target_label, 'predicted']].reset_index(drop=True), use_container_width=True, hide_index=True)


if __name__ == "__main__":
    main()
