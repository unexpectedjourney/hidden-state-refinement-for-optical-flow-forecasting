import torch
import torch.nn as nn
from torch import einsum
from einops import rearrange

from ..encoders import twins_svt_large
from .cnn import BasicEncoder


class StateRefiner(nn.Module):
    def __init__(self, cfg, dim, heads=4, dim_head=128):
        super(StateRefiner, self).__init__()
        self.cfg = cfg
        self.heads = heads
        self.scale = dim_head ** -0.5
        inner_dim = heads * dim_head

        self.to_q = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_k = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.reduce_image_dims = nn.Sequential(
            nn.Conv2d(256, 256, 1, stride=1, padding=0),
            nn.Conv2d(256, 128, 1, stride=1, padding=0),
        )

        self.increase_flow_dims = nn.Sequential(
            nn.Conv2d(2, 128, 1, stride=1, padding=0),
            nn.Conv2d(128, 128, 1, stride=1, padding=0),
        )

        self.reduce_flow_dims = nn.Sequential(
            nn.Conv2d(128, 128, 1, stride=1, padding=0),
            nn.Conv2d(128, 2, 1, stride=1, padding=0),
        )

        self.to_v_flow = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_v_net = nn.Conv2d(dim, inner_dim, 1, bias=False)
        self.to_v_inp = nn.Conv2d(dim, inner_dim, 1, bias=False)

        self.gamma_flow = nn.Parameter(torch.zeros(1))
        self.gamma_net = nn.Parameter(torch.zeros(1))
        self.gamma_inp = nn.Parameter(torch.zeros(1))

        if dim != inner_dim:
            self.project_flow = nn.Conv2d(inner_dim, dim, 1, bias=False)
            self.project_net = nn.Conv2d(inner_dim, dim, 1, bias=False)
            self.project_inp = nn.Conv2d(inner_dim, dim, 1, bias=False)
        else:
            self.project_flow = None
            self.project_net = None
            self.project_inp = None

        if cfg.fnet == 'twins':
            self.feat_encoder = twins_svt_large(pretrained=self.cfg.pretrain)
        elif cfg.fnet == 'basicencoder':
            self.feat_encoder = BasicEncoder(
                output_dim=256, norm_fn='instance')
        else:
            exit()

        self.channel_convertor = nn.Conv2d(
            cfg.encoder_latent_dim,
            cfg.encoder_latent_dim,
            1,
            padding=0,
            bias=False
        )

    def precess_v(self, attn, v, heads, h, w, to_v, project_fn):
        v = to_v(v)
        v = rearrange(v, 'b (h d) x y -> b h (x y) d', h=heads)
        out = einsum('b h i j, b h j d -> b h i d', attn, v)
        out = rearrange(out, 'b h (x y) d -> b (h d) x y', x=h, y=w)

        if project_fn is not None:
            out = project_fn(out)

        return out

    def prepare_image_feats(self, frame_s, frame_t):
        imgs = torch.cat([frame_s, frame_t], dim=0)
        feats = self.feat_encoder(imgs)
        feats = self.channel_convertor(feats)
        B = feats.shape[0] // 2

        feat_s = feats[:B]
        feat_t = feats[B:]
        feat_s = self.reduce_image_dims(feat_s)
        feat_t = self.reduce_image_dims(feat_t)

        return feat_t, feat_s

    def forward(self, prev_frame, curr_frame, prev_flow, prev_net, prev_inp):
        prev_frame, curr_frame = self.prepare_image_feats(
            prev_frame, curr_frame)
        prev_flow_features = self.increase_flow_dims(prev_flow)

        heads, _, _, h, w = self.heads, *prev_frame.shape

        q = self.to_q(curr_frame)
        k = self.to_k(prev_frame)

        q, k = map(
            lambda t: rearrange(t, 'b (h d) x y -> b h x y d', h=heads), (q, k)
        )
        q = self.scale * q
        sim = einsum('b h x y d, b h u v d -> b h x y u v', q, k)
        sim = rearrange(sim, 'b h x y u v -> b h (x y) (u v)')
        attn = sim.softmax(dim=-1)

        out_flow = self.precess_v(
            attn, prev_flow_features, heads, h, w, self.to_v_flow,
            self.project_flow
        )
        out_flow = self.reduce_flow_dims(out_flow)

        out_net = self.precess_v(
            attn, prev_net, heads, h, w, self.to_v_net,
            self.project_net
        )

        out_inp = self.precess_v(
            attn, prev_inp, heads, h, w, self.to_v_inp,
            self.project_inp
        )

        out_flow = prev_flow + self.gamma_flow * out_flow
        out_net = prev_net + self.gamma_net * out_net
        out_inp = prev_inp + self.gamma_inp * out_inp

        return out_flow, out_net, out_inp


class ResidualBlock(nn.Module):
    def __init__(
            self, in_channels, out_channels, stride=1, downsample=None,
            activate=True,
    ):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU()
        )
        self.conv2 = nn.Sequential(
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )
        self.downsample = downsample
        self.relu = nn.ReLU()
        self.out_channels = out_channels
        self.activate = activate

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.conv2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        if self.activate:
            out = self.relu(out)
        return out


class StateMixer(nn.Module):
    def __init__(self):
        super(StateMixer, self).__init__()
        self.flow_out = nn.Sequential(
            nn.Conv2d(4, 128, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(128, 128),
            ResidualBlock(128, 128),
            nn.Conv2d(128, 256, 3, padding=1),
            nn.ReLU(),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            nn.Conv2d(256, 2, 3, padding=1),
            nn.ReLU(),
        )

        self.net_inp_out = nn.Sequential(
            nn.Conv2d(512, 256, 7, padding=3),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256),
            ResidualBlock(256, 256, activate=False),
        )

        self.gamma_flow = nn.Parameter(torch.tensor(0.5))
        self.gamma_net = nn.Parameter(torch.tensor(0.5))
        self.gamma_inp = nn.Parameter(torch.tensor(0.5))

    def forward(
            self,
            flow_init,
            net_init,
            inp_init,
            flow_ref,
            net_ref,
            inp_ref,
    ):
        flow = torch.cat([flow_init, flow_ref], dim=1)
        flow = self.flow_out(flow)

        net_inp = torch.cat([net_init, inp_init, net_ref, inp_ref], dim=1)
        net_inp = self.net_inp_out(net_inp)
        net, inp = torch.split(net_inp, [128, 128], dim=1)
        net = torch.tanh(net)
        inp = torch.relu(inp)

        flow = (1 - self.gamma_flow) * flow_init + self.gamma_flow * flow
        net = (1 - self.gamma_net) * net_init + self.gamma_net * net
        inp = (1 - self.gamma_inp) * inp_init + self.gamma_inp * inp

        return flow, net, inp
