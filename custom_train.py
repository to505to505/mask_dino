from typing import Tuple
from detectron2.config import CfgNode
from detectron2.data import DatasetCatalog, MetadataCatalog
from maskdino.modeling.backbone.swin import D2SwinTransformer
from maskdino.modeling.meta_arch.maskdino_head import MaskDINOHead
from maskdino.modeling.pixel_decoder.maskdino_encoder import MaskDINOEncoder
from maskdino.modeling.matcher import HungarianMatcher
from maskdino.maskdino import MaskDINO
from maskdino.modeling.transformer_decoder import MaskDINODecoder
from maskdino.modeling.criterion import SetCriterion

class MaskDINOCustom(MaskDINO):
    def __init__(
        self
    ):
        # Create Swin config
        cfg = CfgNode()
        cfg.MODEL = CfgNode()
        cfg.MODEL.SWIN = CfgNode()

        # Set Swin parameters
        cfg.MODEL.SWIN.PRETRAIN_IMG_SIZE = 384
        cfg.MODEL.SWIN.PATCH_SIZE = 4
        cfg.MODEL.SWIN.EMBED_DIM = 192
        cfg.MODEL.SWIN.DEPTHS = [2, 2, 18, 2]
        cfg.MODEL.SWIN.NUM_HEADS = [6, 12, 24, 48]
        cfg.MODEL.SWIN.WINDOW_SIZE = 12
        cfg.MODEL.SWIN.MLP_RATIO = 4.0
        cfg.MODEL.SWIN.QKV_BIAS = True
        cfg.MODEL.SWIN.QK_SCALE = None
        cfg.MODEL.SWIN.DROP_RATE = 0.0
        cfg.MODEL.SWIN.ATTN_DROP_RATE = 0.0
        cfg.MODEL.SWIN.DROP_PATH_RATE = 0.3
        cfg.MODEL.SWIN.APE = False
        cfg.MODEL.SWIN.PATCH_NORM = True
        cfg.MODEL.SWIN.USE_CHECKPOINT = False
        cfg.MODEL.SWIN.OUT_FEATURES = ["res2", "res3", "res4", "res5"]




        # Create Swin backbone
        swin_backbone = D2SwinTransformer(cfg, input_shape=None)
        backbone_shape = swin_backbone.output_shape()




        ## pixel decoder
        transformer_dim_feedforward = 2048
        transformer_enc_layers = 6
        conv_dim = 256
        mask_dim = 256
        norm = "GN"
        transformer_in_features = ["res2", "res3", "res4", "res5"]
        common_stride = 4
        num_feature_levels = 4
        total_num_feature_levels = 5
        feature_order = "low2high"

        
        transformer_dropout = 0.0  
        transformer_nheads = 8  

        pixel_decoder = MaskDINOEncoder(input_shape=backbone_shape,
        transformer_dropout=transformer_dropout,
        transformer_nheads=transformer_nheads,
        transformer_dim_feedforward=transformer_dim_feedforward,
        transformer_enc_layers=transformer_enc_layers,
        conv_dim=conv_dim,
        mask_dim=mask_dim,
        norm=norm,
        transformer_in_features=transformer_in_features,
        common_stride=common_stride,
        num_feature_levels=num_feature_levels,
        total_num_feature_levels=total_num_feature_levels,
        feature_order=feature_order,)
        




                # Параметры из секции MODEL.MaskDINO
        hidden_dim = 256
        num_queries = 300
        nheads = 8
        dim_feedforward = 2048
        dec_layers = 9
        dropout = 0.0
        enforce_input_project = False # В YAML ENFORCE_INPUT_PROJ
        two_stage = True
        dn = "seg"
        dn_num = 100
        initialize_box_type = 'bitmask'
        initial_pred = True
        deep_supervision = True # В YAML DEEP_SUPERVISION, соответствует return_intermediate_dec

        # Параметры из секции MODEL.SEM_SEG_HEAD
        num_classes = 80
        mask_dim = 256
        total_num_feature_levels = 5

        # Параметры, для которых используются стандартные значения (т.к. их нет в YAML)
        in_channels = hidden_dim # Входные каналы для декодера - это скрытое измерение модели
        mask_classification = True # MaskDINO по своей сути выполняет классификацию масок
        noise_scale = 0.4  # Типичное стандартное значение для масштаба шума в denoising
        learn_tgt = True # Запросы (targets) в DETR-подобных моделях обычно являются обучаемыми параметрами
        activation = 'relu' # Стандартная функция активации в трансформерах
        dec_n_points = 4 # Стандартное количество точек для Deformable Attention в декодере
        query_dim = 4 # Размерность для представления координат (x, y, w, h) в запросах
        dec_layer_share = False # Слои декодера обычно не разделяют веса
        semantic_ce_loss = False # Включается для panoptic segmentation, у вас PANOPTIC_ON: False


# --- 2. Инициализация объекта decoder ---

        

        transformer_decoder = MaskDINODecoder(
            in_channels=in_channels,
            mask_classification=mask_classification,
            num_classes=num_classes,
            hidden_dim=hidden_dim,
            num_queries=num_queries,
            nheads=nheads,
            dim_feedforward=dim_feedforward,
            dec_layers=dec_layers,
            mask_dim=mask_dim,
            enforce_input_project=enforce_input_project,
            two_stage=two_stage,
            dn=dn,
            noise_scale=noise_scale,
            dn_num=dn_num,
            initialize_box_type=initialize_box_type,
            initial_pred=initial_pred,
            learn_tgt=learn_tgt,
            total_num_feature_levels=total_num_feature_levels,
            dropout=dropout,
            activation=activation,
            dec_n_points=dec_n_points,
            return_intermediate_dec=deep_supervision,
            query_dim=query_dim,
            dec_layer_share=dec_layer_share,
            semantic_ce_loss=semantic_ce_loss,
        )

     

        sem_seg_head = MaskDINOHead(
            input_shape=backbone_shape,
           pixel_decoder = pixel_decoder,
            transformer_predictor=transformer_decoder,
            num_classes=3,
            ignore_value=255,
             loss_weight=1.0, 
            
        )

        # --- 3. Конфигурация и создание Matcher (для функции потерь) ---
        matcher = HungarianMatcher(
            cost_class=4.0,   # из CLASS_WEIGHT
            cost_mask=5.0,    # из MASK_WEIGHT
            cost_dice=5.0,    # из DICE_WEIGHT
            num_points=12544, # из TRAIN_NUM_POINTS
        )
        








        # --- 4. Конфигурация и создание функции потерь (Criterion) ---
        weight_dict = {
            "loss_ce": 4.0,       # из CLASS_WEIGHT
            "loss_mask": 5.0,     # из MASK_WEIGHT
            "loss_dice": 5.0,     # из DICE_WEIGHT
            "loss_box": 5.0,      # из BOX_WEIGHT
            "loss_giou": 2.0,     # из GIOU_WEIGHT
        }



        if two_stage:
            interm_weight_dict = {}
            interm_weight_dict.update({k + f'_interm': v for k, v in weight_dict.items()})
            weight_dict.update(interm_weight_dict)

        weight_dict.update({k + f"_dn": v for k, v in weight_dict.items()})


        if deep_supervision:
            dec_layers = dec_layers
            aux_weight_dict = {}
            for i in range(dec_layers):
                aux_weight_dict.update({k + f"_{i}": v for k, v in weight_dict.items()})
            weight_dict.update(aux_weight_dict)

        criterion = SetCriterion(
            num_classes=3,
            matcher=matcher,
            oversample_ratio=3.0,
            importance_sample_ratio=0.75,
            num_points = 12544,
            weight_dict=weight_dict,
            eos_coef=0.1, 
            losses=["labels", "masks", "boxes"],
            dn="seg",
         
            dn_losses=["labels", "masks", "boxes"],
            panoptic_on=False 
        )

        # --- 5. Инициализация родительского класса MaskDINO со всеми компонентами ---
        # Создаем "пустые" метаданные, так как они обычно приходят из датасета
        metadata = MetadataCatalog.get("right_contrast_v1_train")
    

        # Initialize parent class with our custom backbone
        super().__init__(
            data_loader='coco_instance_lsj',
            backbone=swin_backbone,
            sem_seg_head=sem_seg_head,
            criterion=criterion,
            num_queries=num_queries,
            object_mask_threshold=0.25,
            overlap_threshold=0.8,
            metadata=metadata,
            size_divisibility=32,
            sem_seg_postprocess_before_inference=False,
            pixel_mean=[ 123.675, 116.280, 103.530 ],
            pixel_std=[ 58.395, 57.120, 57.375 ],
            semantic_on=False,
            panoptic_on=False,
            instance_on=True,
            test_topk_per_image=5,
            pano_temp=0,
            
        )