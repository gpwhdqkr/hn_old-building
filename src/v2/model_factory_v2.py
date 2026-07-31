# -*- coding: utf-8 -*-
# ============================================================
# v2 공용 모델 팩토리.
# train/fine_tune/evaluate가 전부 이 파일의 함수만 써서 모델을 만들기 때문에
# "한쪽 스크립트만 구조를 바꿔서 가중치 로드가 실패"하는 사고가 원천 차단됩니다.
#
# 지원 백본 (ARCH env):
#   resnet50 (기본)  : 1차 표준. 기존 resnet 계열 unfreeze 관례(layer4+fc) 계승
#   efficientnet_b0 / efficientnet_b2 : 경량 비교용
#   convnext_tiny    : 텍스처 성능 상한 실험용 — AdamW 강제 (Adam이면 성능 급락 흔함)
# ============================================================

import torch
from torch import nn
from torch.optim import Adam, AdamW
from torchvision.models import (
    convnext_tiny,
    efficientnet_b0,
    efficientnet_b2,
    resnet50,
)
from torchvision.models import (
    ConvNeXt_Tiny_Weights,
    EfficientNet_B0_Weights,
    EfficientNet_B2_Weights,
    ResNet50_Weights,
)

SUPPORTED_ARCHS = ["resnet50", "efficientnet_b0", "efficientnet_b2", "convnext_tiny"]


def create_model(arch, num_classes, use_pretrained_weights):
    """백본을 만들고 분류층을 num_classes 출력으로 교체해서 반환한다."""
    if arch == "resnet50":
        model = resnet50(
            weights=ResNet50_Weights.DEFAULT if use_pretrained_weights else None
        )
        model.fc = nn.Linear(model.fc.in_features, num_classes)

    elif arch == "efficientnet_b0":
        model = efficientnet_b0(
            weights=EfficientNet_B0_Weights.DEFAULT if use_pretrained_weights else None
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif arch == "efficientnet_b2":
        model = efficientnet_b2(
            weights=EfficientNet_B2_Weights.DEFAULT if use_pretrained_weights else None
        )
        model.classifier[1] = nn.Linear(model.classifier[1].in_features, num_classes)

    elif arch == "convnext_tiny":
        model = convnext_tiny(
            weights=ConvNeXt_Tiny_Weights.DEFAULT if use_pretrained_weights else None
        )
        model.classifier[2] = nn.Linear(model.classifier[2].in_features, num_classes)

    else:
        raise ValueError(f"ARCH는 {SUPPORTED_ARCHS} 중 하나여야 합니다: {arch}")

    return model


def head_parameters(model, arch):
    """분류층(head) 파라미터 이터레이터."""
    if arch == "resnet50":
        return model.fc.parameters()
    return model.classifier.parameters()


def freeze_backbone(model, arch):
    """1단계(baseline)용: head를 제외한 전부를 동결한다."""
    for parameter in model.parameters():
        parameter.requires_grad = False
    for parameter in head_parameters(model, arch):
        parameter.requires_grad = True


def unfreeze_for_finetune(model, arch):
    """2단계용: 마지막 특징 블록 + head만 동결 해제. 해제 범위 설명 문자열을 반환한다."""
    for parameter in model.parameters():
        parameter.requires_grad = False

    if arch == "resnet50":
        for parameter in model.layer4.parameters():
            parameter.requires_grad = True
        for parameter in model.fc.parameters():
            parameter.requires_grad = True
        return "layer4 + fc"

    # efficientnet/convnext: 마지막 두 특징 블록 + classifier
    for block in model.features[-2:]:
        for parameter in block.parameters():
            parameter.requires_grad = True
    for parameter in model.classifier.parameters():
        parameter.requires_grad = True
    return "features[-2:] + classifier"


def build_finetune_param_groups(model, arch, lr_backbone, lr_head):
    """2단계용 차등 학습률 param group (백본 쪽은 아주 작게, head는 그보다 크게)."""
    if arch == "resnet50":
        return [
            {"params": model.layer4.parameters(), "lr": lr_backbone},
            {"params": model.fc.parameters(), "lr": lr_head},
        ]
    backbone_params = []
    for block in model.features[-2:]:
        backbone_params.extend(block.parameters())
    return [
        {"params": backbone_params, "lr": lr_backbone},
        {"params": model.classifier.parameters(), "lr": lr_head},
    ]


def set_frozen_modules_eval(model, arch):
    """model.train() 직후 호출: 동결 구간을 평가 상태로 되돌린다.

    가중치뿐 아니라 BatchNorm 이동 통계도 바뀌지 않게 하는 기존 관례.
    (convnext는 LayerNorm이라 통계 오염이 없지만 일관성을 위해 동일 처리)
    """
    if arch == "resnet50":
        model.bn1.eval()
        model.layer1.eval()
        model.layer2.eval()
        model.layer3.eval()
    else:
        for block in model.features[:-2]:
            block.eval()


def create_optimizer(arch, params_or_groups, lr, weight_decay=None):
    """백본별 권장 옵티마이저를 생성한다.

    convnext 계열은 AdamW(+weight_decay)가 사실상 필수 — Adam으로 학습하면
    성능이 급락하는 것이 잘 알려진 함정이라 팩토리에서 강제한다.
    나머지는 기존 관례(Adam, weight_decay 없음)를 유지한다.
    """
    if arch == "convnext_tiny":
        return AdamW(
            params_or_groups,
            lr=lr,
            weight_decay=1e-4 if weight_decay is None else weight_decay,
        )
    return Adam(params_or_groups, lr=lr)


def count_parameters(model):
    """(전체 파라미터 수, 학습 대상 파라미터 수)를 반환한다."""
    total = sum(parameter.numel() for parameter in model.parameters())
    trainable = sum(
        parameter.numel() for parameter in model.parameters() if parameter.requires_grad
    )
    return total, trainable
