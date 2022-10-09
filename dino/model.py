import torch
from torch import nn
from torchvision.models import vit_b_16

class DINOModel(nn.Module):
    def __init__(self) -> None:
        super().__init__()
        self.vit = vit_b_16(weights='DEFAULT')
        # self.vit = torch.nn.Sequential(*(list(self.vit.children())[:-1]))
        self.vit.heads = nn.Identity()
    
    def forward(self, x):
        return self.vit(x)

    @torch.no_grad()
    def update_ema(self, student, l=0.996):
        student_model_params = [param for param in student.parameters()]
        teacher_model_params = [param for param in self.parameters()]
        
        for student_model_param, teacher_model_param in zip(student_model_params, teacher_model_params):
            teacher_model_param = l * teacher_model_param + (1 - l) * student_model_param
        
        return self 

if __name__ == "__main__":
    model = DINOModel()

    rands = torch.randn((4, 3, 224, 224))

    print(model(rands).shape)