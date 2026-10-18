import os
import torch
import argparse
import numpy as np
import pandas as pd
import warnings
warnings.filterwarnings('ignore')
from torchvision import transforms, models
from utils.joint_dataset import CustomDataset
from utils.train_utils import select_model
from tqdm import tqdm
from utils.data_loader import ImageDataset, get_statistics
import torch.nn.functional as F
from torch import nn

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, required=True)
parser.add_argument('--model_name', type=str, default='resnet18', help='Model name')
parser.add_argument('--type', type=str, required=True)
parser.add_argument('--batch_size', type=int, default=32)
parser.add_argument('--learning_rate', type=float, default=1e-4)
parser.add_argument('--num_epochs', type=int, default=10)
args = parser.parse_args()

# Load dataset
if args.dataset == 'PACS_final':
    domain_dirs = {
        'id': './dataset/PACS_final/PACS_final_test_ma',
        'art_painting': './dataset/PACS_final/PACS_MA/art_painting',
        'cartoon': './dataset/PACS_final/PACS_MA/cartoon',
        'sketch': './dataset/PACS_final/PACS_MA/sketch'
    }
elif args.dataset == 'DomainNet':
    domain_dirs = {
        'id': './dataset/DomainNet/DomainNet_test_ma',
        'clipart': './dataset/DomainNet/DomainNet_MA/clipart',
        'infograph': './dataset/DomainNet/DomainNet_MA/infograph',
        'painting': './dataset/DomainNet/DomainNet_MA/painting',
        'quickdraw': './dataset/DomainNet/DomainNet_MA/quickdraw',
        'sketch': './dataset/DomainNet/DomainNet_MA/sketch',
    }
elif args.dataset == 'NICO':
    domain_dirs = {
        # 'id': None,
        'autumn': './dataset/NICO/NICO_MA/autumn',
        'dim': './dataset/NICO/NICO_MA/dim',
        'grass': './dataset/NICO/NICO_MA/grass',
        'outdoor': './dataset/NICO/NICO_MA/outdoor',
        'rock': './dataset/NICO/NICO_MA/rock',
        'water': './dataset/NICO/NICO_MA/water'
    }
elif args.dataset == 'cct':
    domain_dirs = {
        'id': './dataset/cct/cct_in_test_ma',
        'out': './dataset/cct/cct_out_test_ma'
    }
elif args.dataset == 'cifar10':
    domain_dirs = {
        'id': './dataset/cifar10/cifar10_original_test',
        'c': './dataset/cifar10/cifar10_c',
        'nc': './dataset/cifar10/cifar10_nc'
    }



# Define the transform
mean, std, n_classes, inp_size, _ = get_statistics(dataset=args.dataset, type_name=args.type)
transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.RandomCrop(size=(224, 224), padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.RandAugment(),
    # transforms.ConvertImageDtype(),
    transforms.ToTensor(),
    transforms.Normalize(mean=mean, std=std)
])
train_root_dir = os.path.join('./dataset', args.dataset, f'{args.dataset}_{args.type}')
train_dataset = CustomDataset(train_root_dir, transform=transform)
train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=args.batch_size, shuffle=True)

test_datasets = {}
for domain, domain_dir in domain_dirs.items():
    if domain_dir is not None:
        test_dataset = CustomDataset(domain_dir, transform=transform)
        test_datasets[domain] = test_dataset
        
        # Get the number of classes
        num_classes = len(set(train_dataset.labels))
    else:
        test_datasets[domain] = None
test_dataloaders = {domain: torch.utils.data.DataLoader(test_dataset, batch_size=args.batch_size, shuffle=False) for domain, test_dataset in test_datasets.items()}


# Training
def train(model, criterion, optimizer, train_loader, device):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def test(model, criterion, test_dataloader, device):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            running_loss += loss.item()
            _, preds = torch.max(outputs, 1)
            corrects += torch.sum(preds == labels).item()
            total += labels.size(0)
    return running_loss / len(test_dataloader), corrects / total



# Training
def clip_train(model, criterion, optimizer, train_loader, device, text_class_tokens):
    model.train()
    running_loss = 0.0
    for inputs, labels in tqdm(train_loader):
        inputs, labels = inputs.to(device), labels.to(device)
        optimizer.zero_grad()
        outputs = model(inputs, text_class_tokens)
        # logits_per_image =outputs['logit_scale'] * outputs['image_features'] @ outputs['text_features'].T
        # loss = F.cross_entropy(logits_per_image, labels)
        # text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
        # top_probs, top_labels = text_probs.topk(1, dim=-1)
        # total += labels.size(0)
        # correct += torch.sum(top_labels == labels.unsqueeze(1)).item()
        loss = criterion(*outputs, labels=labels)
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    return running_loss / len(train_loader)

def clip_test(model, criterion, test_dataloader, device, text_class_tokens):
    model.eval()
    running_loss = 0.0
    corrects = 0
    total = 0
    with torch.no_grad():
        for inputs, labels in test_dataloader:
            inputs, labels = inputs.to(device), labels.to(device)
            image_features = model.encode_image(inputs)
            text_features = model.encode_text(text_class_tokens)
            image_features /= image_features.norm(dim=-1, keepdim=True)
            text_features /= text_features.norm(dim=-1, keepdim=True)
            text_probs = (100.0 * image_features @ text_features.T).softmax(dim=-1)
            top_probs, top_labels = text_probs.topk(1, dim=-1)
            total += labels.size(0)
            corrects += torch.sum(top_labels == labels.unsqueeze(1)).item()
    # return running_loss / len(test_dataloader), corrects / total
    return None, corrects / total
                
def main():
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    train_root_dir = os.path.join('./dataset', args.dataset, f'{args.dataset}_{args.type}')
    if "clip" in args.model_name:
        model, pretrain_train_transform, pretrain_val_transform, tokenizer, criterion = select_model(args.model_name, args.dataset)
        model = model.to(device)
        criterion = ClipLoss().to(device)
        prompt_template = 'this is a photo of a '
        exposed_classes = list(os.listdir(train_root_dir))
        text_class_prompts = [prompt_template+cla for cla in exposed_classes]
        text_class_tokens = tokenizer(text_class_prompts).to(device)
    else:
        model = select_model(args.model_name, dataset=args.dataset, num_classes=num_classes).to(device)
        criterion = torch.nn.CrossEntropyLoss()
    
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate)
    
    eval_results = {
        'in_domain_intermediate': [],
        'ood_domain_intermediate': {}
    }
    
    # Create csv save directory
    save_dir = os.path.join('./results', args.dataset, "joint", args.type)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    csv_path = os.path.join(save_dir, f'{args.model_name}.csv')
    with open(csv_path, 'w') as f:
        f.write('Epoch,In Domain,OOD\n')
    
    for epoch in range(args.num_epochs):
        print(f"Starting epoch {epoch+1}/{args.num_epochs}")
        if "clip" in args.model_name:
            train_loss = clip_train(model, criterion, optimizer, train_dataloader, device, text_class_tokens)
        else:
            train_loss = train(model, criterion, optimizer, train_dataloader, device)
        # Evaluate
        for domain, test_dataloader in tqdm(test_dataloaders.items()):
            if test_dataloader is not None:
                if "clip" in args.model_name:
                    _, accuracy = clip_test(model, criterion, test_dataloader, device, text_class_tokens)
                else:
                    test_loss, accuracy = test(model, criterion, test_dataloader, device)
                if domain == 'id':
                    eval_results['in_domain_intermediate'].append(accuracy)
                else:
                    if domain not in eval_results['ood_domain_intermediate']:
                        eval_results['ood_domain_intermediate'][domain] = []
                    eval_results['ood_domain_intermediate'][domain].append(accuracy)
        
        # Print intermediate results
        ood_average_current = np.mean(
            [eval_results['ood_domain_intermediate'][domain][-1] for domain in eval_results['ood_domain_intermediate']],
            axis=0
        )
        print(f"Epoch {epoch+1}/{args.num_epochs}, In Domain Accuracy: {eval_results['in_domain_intermediate'][-1]:.4f}")
        print(f"Epoch {epoch+1}/{args.num_epochs}, OOD Domain Accuracy: {ood_average_current:.4f}")

        # Save intermediate results
        epoch_data = {
            'Epoch': epoch,
            'In Domain': eval_results['in_domain_intermediate'][-1],
            'OOD': ood_average_current
        }
        df_epoch = pd.DataFrame(epoch_data, index=[0])
        df_epoch.to_csv(csv_path, mode='a', header=False, index=False)
    
    # Save average results
    in_domain_average = np.mean(eval_results['in_domain_intermediate'], axis=0)
    ood_average = np.mean(
        [eval_results['ood_domain_intermediate'][domain] for domain in eval_results['ood_domain_intermediate']],
        axis=0
    )
    ood_average = np.mean(ood_average)
    df_average = pd.DataFrame({
        'Epoch': 'Average',
        'In Domain': in_domain_average,
        'OOD': ood_average
    }, index=[0])
    df_average.to_csv(csv_path, mode='a', header=False, index=False)
    
    train_root_dir = os.path.join('./dataset', args.dataset, f'{args.dataset}_{args.type}')
    torch.save({
        'model_state_dict': model.state_dict(),
        'exposed_classes': list(os.listdir(train_root_dir))
        }, f'./real_ckpt/{args.model_name}_{args.dataset}.pth')
    
    
class ClipLoss(nn.Module):

    def __init__(
            self,
            local_loss=False,
            gather_with_grad=False,
            cache_labels=True,
            rank=0,
            world_size=1,
            use_horovod=False,
    ):
        super().__init__()
        self.local_loss = local_loss
        self.gather_with_grad = gather_with_grad
        self.cache_labels = cache_labels
        self.rank = rank
        self.world_size = world_size
        self.use_horovod = use_horovod

        # cache state
        self.prev_num_logits = 0
        self.labels = {}

    def get_ground_truth(self, device, num_logits) -> torch.Tensor:
        # calculated ground-truth and cache if enabled
        if self.prev_num_logits != num_logits or device not in self.labels:
            labels = torch.arange(num_logits, device=device, dtype=torch.long)
            if self.world_size > 1 and self.local_loss:
                labels = labels + num_logits * self.rank
            if self.cache_labels:
                self.labels[device] = labels
                self.prev_num_logits = num_logits
        else:
            labels = self.labels[device]
        return labels

    def get_logits(self, image_features, text_features, logit_scale):
        if self.world_size > 1:
            all_image_features, all_text_features = gather_features(
                image_features, text_features,
                self.local_loss, self.gather_with_grad, self.rank, self.world_size, self.use_horovod)

            if self.local_loss:
                logits_per_image = logit_scale * image_features @ all_text_features.T
                logits_per_text = logit_scale * text_features @ all_image_features.T
            else:
                logits_per_image = logit_scale * all_image_features @ all_text_features.T
                logits_per_text = logits_per_image.T
        else:
            logits_per_image = logit_scale * image_features @ text_features.T
            logits_per_text = logit_scale * text_features @ image_features.T
        
        return logits_per_image, logits_per_text

    def forward(self, image_features, text_features, logit_scale,labels, output_dict=False):
        device = image_features.device
        logits_per_image, logits_per_text = self.get_logits(image_features, text_features, logit_scale)

        # labels = self.get_ground_truth(device, logits_per_image.shape[0])
        total_loss = F.cross_entropy(logits_per_image, labels)
            

        # return {"contrastive_loss": total_loss} if output_dict else total_loss
        return total_loss


if __name__ == '__main__':
    main()