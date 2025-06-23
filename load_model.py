import torch 
from custom_maskdino import MaskDINOCustom
from collections import OrderedDict





def load_model(    model: MaskDINOCustom,
    actual_state_dict: OrderedDict,
):
    


    final_state_dict_to_load = OrderedDict()
    model_expected_keys = set(model.state_dict().keys()) # Используем set для быстрой проверки in
    num_decoder_layers_in_model = model.sem_seg_head.predictor.num_layers # или dec_layers из конфига/кода

    # --- Шаг 1: Извлечь "чистые" веса MLP для bbox_embed из скачанного файла ---
    # Мы знаем, что в файле они имеют формат "sem_seg_head.predictor.decoder.bbox_embed.X..."
    # Возьмем их из ".0." (первый элемент ModuleList)
    source_bbox_mlp_weights = OrderedDict() # {".layers.N.weight": tensor, ...}
    file_bbox_embed_source_prefix = "sem_seg_head.predictor.decoder.bbox_embed.0" # Источник
    found_bbox_weights_in_file = False

    for file_key, value in actual_state_dict.items():
        if file_key.startswith(file_bbox_embed_source_prefix):
            # file_key: "sem_seg_head.predictor.decoder.bbox_embed.0.layers.N.weight"
            # suffix: ".layers.N.weight"
            suffix = file_key[len(file_bbox_embed_source_prefix):]
            source_bbox_mlp_weights[suffix] = value
            found_bbox_weights_in_file = True

    if found_bbox_weights_in_file:
        print(f"Найдены веса bbox_embed в файле по шаблону '{file_bbox_embed_source_prefix}...'. "
            f"Количество извлеченных параметров для MLP: {len(source_bbox_mlp_weights)}")
    else:
        print("ВНИМАНИЕ: Не удалось найти веса для bbox_embed в скачанном state_dict по ожидаемому шаблону "
            f"'{file_bbox_embed_source_prefix}...'. Часть модели bbox_embed останется неинициализированной.")

    # --- Шаг 2: Если веса bbox_embed найдены, "размножить" их под все ожидаемые моделью имена ---
    if found_bbox_weights_in_file and len(source_bbox_mlp_weights) > 0:
        # 2а. Для ключей типа "sem_seg_head.predictor._bbox_embed..."
        model_underscore_prefix = "sem_seg_head.predictor._bbox_embed"
        for suffix, value in source_bbox_mlp_weights.items():
            model_key = model_underscore_prefix + suffix
            if model_key in model_expected_keys:
                final_state_dict_to_load[model_key] = value
                # print(f"Mapping to _bbox_embed: {model_key}")
            # else:
                # print(f"Предупреждение: модель не ожидает ключ {model_key} для _bbox_embed")

        # 2б. Для ключей типа "sem_seg_head.predictor.bbox_embed.X..." (прямой ModuleList)
        model_direct_list_prefix = "sem_seg_head.predictor.bbox_embed"
        for i in range(num_decoder_layers_in_model):
            for suffix, value in source_bbox_mlp_weights.items(): # Используем те же source_bbox_mlp_weights
                model_key = f"{model_direct_list_prefix}.{i}{suffix}"
                if model_key in model_expected_keys:
                    final_state_dict_to_load[model_key] = value
                    # print(f"Mapping to direct bbox_embed.{i}: {model_key}")
                # else:
                    # print(f"Предупреждение: модель не ожидает ключ {model_key} для bbox_embed.{i}")

        # 2в. Для ключей типа "sem_seg_head.predictor.decoder.bbox_embed.X..." (ModuleList внутри decoder)
        # Эти ключи УЖЕ ЕСТЬ в actual_state_dict, поэтому мы их просто скопируем в Шаге 3,
        # если они не были перехвачены как часть source_bbox_mlp_weights (что не должно произойти
        # при правильном file_bbox_embed_source_prefix).
        # Однако, если бы мы их не копировали отдельно, то они бы тоже заполнились здесь.
        model_decoder_list_prefix = "sem_seg_head.predictor.decoder.bbox_embed"
        for i in range(num_decoder_layers_in_model):
            for suffix, value in source_bbox_mlp_weights.items(): # Используем те же source_bbox_mlp_weights
                model_key = f"{model_decoder_list_prefix}.{i}{suffix}"
                if model_key in model_expected_keys:
                    if model_key not in final_state_dict_to_load: # Не перезаписывать, если уже есть
                        final_state_dict_to_load[model_key] = value
                        # print(f"Mapping to decoder.bbox_embed.{i}: {model_key}")
                # else:
                    # print(f"Предупреждение: модель не ожидает ключ {model_key} для decoder.bbox_embed.{i}")
    else:
        print("Веса для bbox_embed не были успешно извлечены из файла, эта часть модели (все три 'вида' bbox_embed) "
            "останется с начальной инициализацией или будет в missing_keys.")


    # --- Шаг 3: Скопировать остальные ключи ---
    # Копируем все ключи из actual_state_dict, которые также есть в model_expected_keys,
    # и которые еще не были добавлены в final_state_dict_to_load (чтобы не перезаписать нашу работу с bbox_embed).

    copied_other_keys_count = 0
    skipped_due_to_shape_mismatch = []

    for file_key, value in actual_state_dict.items():
        # Пропускаем ключи bbox_embed из файла, так как мы их уже специфично обработали
        # через source_bbox_mlp_weights и размножили.
        # Этот if гарантирует, что мы не пытаемся их копировать еще раз "как есть".
        is_original_bbox_embed_key_from_file = False
        for i in range(num_decoder_layers_in_model): # Проверяем все возможные индексы из файла
            if file_key.startswith(f"sem_seg_head.predictor.decoder.bbox_embed.{i}"):
                is_original_bbox_embed_key_from_file = True
                break
        if is_original_bbox_embed_key_from_file:
            continue # Мы уже обработали эти веса через source_bbox_mlp_weights

        # Если ключ из файла ожидается моделью и еще не был добавлен
        if file_key in model_expected_keys:
            if file_key not in final_state_dict_to_load:
                # Проверка на совпадение размеров для criterion.empty_weight
                if file_key == "criterion.empty_weight":
                    model_tensor_shape = model.state_dict()[file_key].shape
                    if model_tensor_shape != value.shape:
                        print(f"Пропуск '{file_key}' из-за несовпадения размера (файл: {value.shape}, модель: {model_tensor_shape})")
                        skipped_due_to_shape_mismatch.append(file_key)
                        continue
                
                final_state_dict_to_load[file_key] = value
                copied_other_keys_count += 1
        # else:
            # Ключ из файла не ожидается моделью под таким же именем,
            # он попадет в unexpected_keys, если мы его не переименуем.
            # Для простоты, пока не добавляем сложную логику переименования для остальных.
            # print(f"Информация: ключ из файла '{file_key}' не найден в ожидаемых моделью ключах под тем же именем.")
            pass


    print(f"Скопировано {copied_other_keys_count} других совпадающих ключей.")
    if skipped_due_to_shape_mismatch:
        print(f"Пропущены ключи из-за несовпадения размеров: {skipped_due_to_shape_mismatch}")


    # --- Шаг 4: Загрузка ---
    print(f"\nПопытка загрузить {len(final_state_dict_to_load)} ключей в модель с strict=False...")
    load_result = model.load_state_dict(final_state_dict_to_load, strict=False)

    # --- Анализ результата ---
    print("\n--- Результат model.load_state_dict ---")
    if load_result.missing_keys:
        print(f"ОТСУТСТВУЮЩИЕ КЛЮЧИ ({len(load_result.missing_keys)}) (эти слои в вашей текущей модели НЕ были загружены):")
        for key in load_result.missing_keys:
            print(f"  - {key}")
            if "bbox_embed" in key:
                print("    ^ (Если это ключ bbox_embed, возможно, исходные веса не нашлись или не размножились корректно)")
            elif "query_feat" in key:
                print("    ^ (Этот ключ отсутствует в скачанном файле)")
    else:
        print("Отсутствующих ключей нет.")

    if load_result.unexpected_keys:
        print(f"\nНЕОЖИДАННЫE КЛЮЧИ ({len(load_result.unexpected_keys)}) (эти слои были в файле, но НЕ в вашей текущей модели или не подошли):")
        for key in load_result.unexpected_keys:
            print(f"  - {key}")
            if "criterion.empty_weight" in key and key in skipped_due_to_shape_mismatch:
                pass # Уже сообщили
            elif "bbox_embed" in key:
                print("    ^ (Если это ключ bbox_embed из файла, он должен был быть обработан. Появление здесь - ошибка в логике.)")
    else:
        print("Неожиданных ключей нет.")
    return model, load_result