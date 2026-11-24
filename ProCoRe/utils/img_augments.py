import torchvision.transforms.functional as TF

def rotate(x):
    img1 = x.get("image", None)
    img2 = TF.rotate(img1, 90)
    img3 = TF.rotate(img1, 180)
    img4 = TF.rotate(img1, 270)
    x["image"] = [img1, img2, img3, img4]
    return x

def shuffle(self, x, order=[0,1,2,3]):
    _, height, width = x.shape
    # For easy implementation, assume image width and height are even numbers.
    center_x = width // 2
    center_y = height // 2
    x1 = torch.zeros_like(x)
    
    x_split = [x[:, :center_y, :center_x], x[:, :center_y, center_x:], x[:, center_y:, :center_x], x[:, center_y:, center_x:]]
    
    x1[:, :center_y, :center_x] = x_split[order[0]]
    x1[:, :center_y, center_x:] = x_split[order[1]]
    x1[:, center_y:, :center_x] = x_split[order[2]]
    x1[:, center_y:, center_x:] = x_split[order[3]]
    
    return x1

def permute(x):
    img1 = x.get("image", None)
    '''
    rotate reverse clock wise
    1 3
    0 2
    '''
    img2 = shuffle(img1, order=[1, 3, 0, 2])
    '''
    rotate clock wise
    2 0
    3 1
    '''
    img3 = shuffle(img1, order=[2, 0, 3, 1])
    
    '''
    cross flip
    3 2
    1 0
    '''
    img4 = shuffle(img1, order=[3, 2, 1, 0])
    x["image"] = [img1, img2, img3, img4]
    return x

def gaussian_noise(x):
    img1 = x.get("image", None)
    img2 = torch.randn(img1.shape)
    img3 = torch.randn(img1.shape)
    img4 = torch.randn(img1.shape)
    x["image"] = [img1, img2, img3, img4]
    return x

def get_augment(hard_type):
    
    if hard_type == "rotate":
        return rotate()
    elif hard_type == "permute":
        return permute()
    elif hard_type == "noise":
        return gaussian_noise()
    else:
        raise NotImplementedError("You must choose one of [rotate, permute, noise]")
    return 