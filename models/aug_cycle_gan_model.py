import torch
import itertools
from util.image_pool import ImagePool
from .base_model import BaseModel
from . import networks


class AugCycleGANModel(BaseModel):

    @staticmethod
    def modify_commandline_options(parser, is_train=True):
        parser.set_defaults(no_dropout=True)  # default CycleGAN did not use dropout
        parser.add_argument('--nlatent', type=int, default=16, help='# of latent code dimensions. Used only for stochastic models, e.g. cycle_ali')
        parser.add_argument('--nef', type=int, default=32, help='# of encoder filters in first conv layer')
        if is_train:
            parser.add_argument('--lambda_A', type=float, default=10.0, help='weight for cycle loss (A -> B -> A)')
            parser.add_argument('--lambda_B', type=float, default=10.0, help='weight for cycle loss (B -> A -> B)')
        return parser

    def __init__(self, opt):
        BaseModel.__init__(self, opt)
        self.loss_names = ['G_A','G_z_A','cycle_A','cycle_z_A','G_B','G_z_B','cycle_B','cycle_z_B',]
        visual_names_A = ['real_A', 'fake_B', 'rec_A',]
        visual_names_B = ['real_B', 'fake_A', 'rec_B',]
        self.visual_names = visual_names_A + visual_names_B 
        if self.isTrain:
            self.model_names = ['G_A', 'G_B', 'D_A', 'D_B','E_A','E_B',]
        else:
            self.model_names = ['G_A', 'G_B','E_A','E_B',]

        self.netG_A = networks.define_stochastic_G(nlatent=opt.nlatent, input_nc=opt.input_nc,
                                                     output_nc=opt.output_nc, ngf=opt.ngf,
                                                     norm=opt.norm, gpu_ids=opt.gpu_ids)

        self.netG_B = networks.define_stochastic_G(nlatent=opt.nlatent, input_nc=opt.output_nc,
                                                     output_nc=opt.input_nc, ngf=opt.ngf,
                                                     norm=opt.norm, gpu_ids=opt.gpu_ids)

        enc_input_nc = opt.input_nc + opt.output_nc

        self.netE_A = networks.define_E(nlatent=opt.nlatent, input_nc=enc_input_nc,
                                        nef=opt.nef, norm='batch', gpu_ids=opt.gpu_ids)

        self.netE_B = networks.define_E(nlatent=opt.nlatent, input_nc=enc_input_nc,
                                        nef=opt.nef, norm='batch', gpu_ids=opt.gpu_ids)

        if self.isTrain:  # define discriminators
            self.netD_A = networks.define_D(opt.output_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD_B = networks.define_D(opt.input_nc, opt.ndf, opt.netD,
                                            opt.n_layers_D, opt.norm, opt.init_type, opt.init_gain, self.gpu_ids)

            self.netD_z_A = networks.define_LAT_D(nlatent=opt.nlatent, ndf=opt.ndf,
                                                gpu_ids=opt.gpu_ids)

            self.netD_z_B = networks.define_LAT_D(nlatent=opt.nlatent, ndf=opt.ndf,
                                                gpu_ids=opt.gpu_ids)

        if self.isTrain:
            # define loss functions
            self.criterionGAN = networks.GANLoss(opt.gan_mode).to(self.device)  # define GAN loss.
            self.criterionCycle = torch.nn.L1Loss()
            self.criterionIdt = torch.nn.L1Loss()
            # initialize optimizers; schedulers will be automatically created by function <BaseModel.setup>.
            self.optimizer_G = torch.optim.Adam(itertools.chain(self.netG_A.parameters(), self.netG_B.parameters(), self.netE_A.parameters(), self.netE_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizer_D = torch.optim.Adam(itertools.chain(self.netD_A.parameters(), self.netD_B.parameters(), self.netD_z_A.parameters(), self.netD_z_B.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
            self.optimizers.append(self.optimizer_G)
            self.optimizers.append(self.optimizer_D)

    def set_input(self, input):
        """Unpack input data from the dataloader and perform necessary pre-processing steps.

        Parameters:
            input (dict): include the data itself and its metadata information.

        The option 'direction' can be used to swap domain A and domain B.
        """
        AtoB = self.opt.direction == 'AtoB'
        self.real_A = input['A' if AtoB else 'B'].to(self.device)
        self.real_B = input['B' if AtoB else 'A'].to(self.device)
        self.prior_z_A = torch.randn((self.real_B.shape[0], self.opt.nlatent, 1, 1)).to(self.device)
        self.prior_z_B = torch.randn((self.real_A.shape[0], self.opt.nlatent, 1, 1)).to(self.device)
        self.image_paths = input['A_paths' if AtoB else 'B_paths']

    def forward(self):
        """Run forward pass; called by both functions <optimize_parameters> and <test>."""
        self.fake_B = self.netG_A(self.real_A, self.prior_z_B) 
        self.mu_z_realA = self.netE_A(torch.cat((self.real_A, self.fake_B),1))
        self.rec_A = self.netG_B(self.fake_B, self.mu_z_realA)
        self.rec_prior_z_B = self.netE_B(torch.cat((self.real_A, self.fake_B),1))
        
        self.fake_A = self.netG_B(self.real_B, self.prior_z_A) 
        self.mu_z_realB = self.netE_B(torch.cat((self.fake_A, self.real_B),1))
        self.rec_B = self.netG_B(self.fake_A, self.mu_z_realB)
        self.rec_prior_z_A = self.netE_B(torch.cat((self.fake_A, self.real_B),1))

    def backward_D_basic(self, netD, real, fake):
        """Calculate GAN loss for the discriminator

        Parameters:
            netD (network)      -- the discriminator D
            real (tensor array) -- real images
            fake (tensor array) -- images generated by a generator

        Return the discriminator loss.
        We also call loss_D.backward() to calculate the gradients.
        """
        # Real
        pred_real = netD(real)
        loss_D_real = self.criterionGAN(pred_real, True)
        # Fake
        pred_fake = netD(fake.detach())
        loss_D_fake = self.criterionGAN(pred_fake, False)
        # Combined loss and calculate gradients
        loss_D = (loss_D_real + loss_D_fake) * 0.5
        loss_D.backward()
        return loss_D

    def backward_D_A(self):
        """Calculate GAN loss for discriminator D_A"""
        self.loss_D_A = self.backward_D_basic(self.netD_A, self.real_B, self.fake_B)
        self.loss_D_z_A = self.backward_D_basic(self.netD_z_A, self.prior_z_A, self.mu_z_realA)

    def backward_D_B(self):
        """Calculate GAN loss for discriminator D_B"""
        self.loss_D_B = self.backward_D_basic(self.netD_B, self.real_A, self.fake_A)
        self.loss_D_z_B = self.backward_D_basic(self.netD_z_B, self.prior_z_B, self.mu_z_realB)

    def backward_G(self):
        """Calculate the loss for generators G_A and G_B"""
        lambda_A = self.opt.lambda_A
        lambda_B = self.opt.lambda_B

        # GAN loss D_A(G_A(A))
        self.loss_G_A = self.criterionGAN(self.netD_A(self.fake_B), True)
        self.loss_G_z_A = self.criterionGAN(self.netD_z_A(self.mu_z_realA), True)
        # GAN loss D_B(G_B(B))
        self.loss_G_B = self.criterionGAN(self.netD_B(self.fake_A), True)
        self.loss_G_z_B = self.criterionGAN(self.netD_z_B(self.mu_z_realB), True)
        # Forward cycle loss || G_B(G_A(A)) - A||
        self.loss_cycle_A = self.criterionCycle(self.rec_A, self.real_A) * lambda_A
        self.loss_cycle_z_A = self.criterionCycle(self.rec_prior_z_B, self.prior_z_B) * lambda_A
        # Backward cycle loss || G_A(G_B(B)) - B||
        self.loss_cycle_B = self.criterionCycle(self.rec_B, self.real_B) * lambda_B
        self.loss_cycle_z_B = self.criterionCycle(self.rec_prior_z_A, self.prior_z_A) * lambda_B
        # combined loss and calculate gradients
        self.loss_G = self.loss_G_A + self.loss_G_B + self.loss_cycle_A + self.loss_cycle_B \
        + self.loss_G_z_A + self.loss_G_z_B + self.loss_cycle_z_A + self.loss_cycle_z_B
        self.loss_G.backward()

    def optimize_parameters(self):
        """Calculate losses, gradients, and update network weights; called in every training iteration"""
        # forward
        self.forward()      # compute fake images and reconstruction images.
        # G_A and G_B
        self.set_requires_grad([self.netD_A, self.netD_B], False)  # Ds require no gradients when optimizing Gs
        self.optimizer_G.zero_grad()  # set G_A and G_B's gradients to zero
        self.backward_G()             # calculate gradients for G_A and G_B
        self.optimizer_G.step()       # update G_A and G_B's weights
        # D_A and D_B
        self.set_requires_grad([self.netD_A, self.netD_B], True)
        self.optimizer_D.zero_grad()   # set D_A and D_B's gradients to zero
        self.backward_D_A()      # calculate gradients for D_A
        self.backward_D_B()      # calculate graidents for D_B
        self.optimizer_D.step()  # update D_A and D_B's weights
