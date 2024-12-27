import locate_object
import torch


if __name__ == '__main__':

    # locate_object.run()
    a = torch.zeros(1, 2)
    a[0, 1] = 5
    print(a)