import torch
import argparse
# import yaml
# import yaml_utils
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import datasets, transforms,utils
from torch.utils.data import DataLoader
from gen_models_pytorch.resnet import Generator
from dis_models_pytorch.snresnet import Discriminator
import torchvision

device = torch.device('cuda:1' if torch.cuda.is_available() else 'cpu')


def main():


    parser = argparse.ArgumentParser()
    parser.add_argument('--batch_size', type=int, default=2)
    parser.add_argument('--lr', type=float, default=2e-4)
    parser.add_argument('--loss', type=str, default='hinge')
    parser.add_argument('--checkpoint_dir', type=str, default='checkpoints')

    parser.add_argument('--model', type=str, default='resnet')
    parser.add_argument('--path', type=str, default=r'H:\Dataset\flowers17\train')
    # parser.add_argument('--batch', type=int, default=8)
    parser.add_argument('--iter', type=int, default=200000)
    parser.add_argument('--n_class', type=int, default=10)


    args = parser.parse_args()

    dataset = iter(sample_data(args.path, args.batch_size))

    Z_dim = 128
    # number of updates to discriminator for every update to generator
    disc_iters = 3

    discriminator = Discriminator(n_class=args.n_class).to(device)
    generator = Generator(Z_dim,n_class=args.n_class).to(device)

    # because the spectral normalization module creates parameters that don't require gradients (u and v), we don't want to
    # optimize these using sgd. We only let the optimizer operate on parameters that _do_ require gradients
    # TODO: replace Parameters with buffers, which aren't returned from .parameters() method.
    optim_disc = optim.Adam(discriminator.parameters(), lr=args.lr, betas=(0.0, 0.9))
    optim_gen = optim.Adam(generator.parameters(), lr=args.lr, betas=(0.0, 0.9))

    # use an exponentially decaying learning rate
    scheduler_d = optim.lr_scheduler.ExponentialLR(optim_disc, gamma=0.99)
    scheduler_g = optim.lr_scheduler.ExponentialLR(optim_gen, gamma=0.99)
    pbar = tqdm(range(args.iter), dynamic_ncols=True)

    for i in pbar:
        discriminator.zero_grad()
        # real_image, label = next(dataset)
        # b_size = real_image.size(0)
        # real_image = real_image.to(device)
        # label = label.to(device)

        # update discriminator
        requires_grad(generator, False)
        requires_grad(discriminator, True)
        b_size = 0
        for _ in range(disc_iters):
            real_image, label = next(dataset)
            real_image = real_image.to(device)
            b_size = real_image.size(0)
            z = torch.randn(b_size, Z_dim).to(device)
            label = label.to(device)
            optim_disc.zero_grad()
            optim_gen.zero_grad()

            disc_loss = -discriminator(real_image,label).mean() + discriminator(generator(z,label),label).mean()

            # if args.loss == 'hinge':
            #     disc_loss = nn.ReLU()(1.0 - discriminator(data)).mean() + nn.ReLU()(1.0 + discriminator(generator(z))).mean()
            # elif args.loss == 'wasserstein':
            #     disc_loss = -discriminator(data).mean() + discriminator(generator(z)).mean()
            # else:
            #     disc_loss = nn.BCEWithLogitsLoss()(discriminator(data), Variable(torch.ones(args.batch_size, 1).cuda())) + \
            #         nn.BCEWithLogitsLoss()(discriminator(generator(z)), Variable(torch.zeros(args.batch_size, 1).cuda()))

            disc_loss.backward()
            optim_disc.step()

        # optim_disc.zero_grad()
        optim_gen.zero_grad()

        requires_grad(generator, True)
        requires_grad(discriminator,False )

        z = torch.randn(args.batch_size, Z_dim).to(device)
        gen_loss = -discriminator(generator(z,label),label).mean()
        gen_loss.backward()
        optim_gen.step()

        if i%10000 == 0:
            scheduler_d.step()
            scheduler_g.step()

        if (i + 1) % 100 == 0:

            generator.train(False)
            z = torch.randn(args.n_class, Z_dim).to(device)
            input_class = torch.arange(args.n_class).long().to(device)
            fake_image = generator(z, input_class)
            generator.train(True)
            utils.save_image(
                fake_image.cpu().data,
                f'sample/{str(i + 1).zfill(7)}.png',
                nrow=args.n_class,
                normalize=True,
                range=(-1, 1),
            )

        if (i + 1) % 5000 == 0:
            no = str(i + 1).zfill(7)
            torch.save(generator.state_dict(), f'checkpoint/generator_{no}.pt')
            torch.save(discriminator.state_dict(), f'checkpoint/discriminator_{no}.pt')
            torch.save(optim_gen.state_dict(), f'checkpoint/gen_optimizer_{no}.pt')
            torch.save(optim_disc.state_dict(), f'checkpoint/dis_optimizer_{no}.pt')

        pbar.set_description(
            (f'{i + 1}; G: {gen_loss:.5f};' f' D: {disc_loss:.5f}')
        )


def sample_data(path, batch_size):
    # dataset = datasets.ImageFolder(path, transform=transform)
    dataset = torchvision.datasets.STL10(path,transform=transform)
    loader = DataLoader(dataset, shuffle=True, batch_size=batch_size, num_workers=4)
    loader = iter(loader)

    while True:
        try:
            yield next(loader)

        except StopIteration:
            loader = DataLoader(
                dataset, shuffle=True, batch_size=batch_size, num_workers=4
            )
            loader = iter(loader)
            yield next(loader)

transform = transforms.Compose(
    [
        transforms.Resize((128,128)),
        # transforms.CenterCrop(128),
        # transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        # transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ]
)

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag



if __name__ == '__main__':
    with torch.cuda.device(1):

        main()





