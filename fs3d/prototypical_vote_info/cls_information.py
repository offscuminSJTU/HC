import torch

def feature_norm(feature):
    features_norm = torch.norm(feature, p=2, dim=1)
    feature = feature.div(features_norm.unsqueeze(1))
    return feature

def cls_gather_grad(x, pts_semantic_masks, pts_instance_masks, points_xyzs, seed_indices, class_names=None, background_flag=17):

    points = x['fp_xyz'][-1]
    features = x['fp_features'][-1] # (2*b, 128, 4096)
    # 注意，提取作为prototype时, 这里切断梯度了, 但想要计算cl_loss, 不应该切断梯度
    fp_indices = x['fp_indices'][-1]

    if class_names[0] == 'cabinet':
        num_p = 40000
    else:
        num_p = 20000

    batch_size = features.shape[0]
    batch_list = []
    for ii in range(batch_size):
        feature = features[ii]
        pts_semantic_mask = pts_semantic_masks[ii]
        pts_instance_mask = pts_instance_masks[ii]
        # print('pts_semantic_mask: ', torch.unique(pts_semantic_mask)) # batch中的不同样本含有不同的semantic_mask

        new_dict = {}
        unique_pts_instance_mask = torch.unique(pts_instance_mask)
        binary_pts_instance_mask = pts_instance_mask.new(num_p, ).zero_()
        binary_pts_instance_mask[fp_indices[ii]] = 1

        index = binary_pts_instance_mask.new_ones(fp_indices.shape[-1])
        index = index.cumsum(dim=0) - 1
        binary_index = pts_instance_mask.new(num_p, ).zero_()
        binary_index[fp_indices[ii]] = index

        for i in unique_pts_instance_mask:
            indices = torch.nonzero(pts_instance_mask == i, as_tuple=False).squeeze(-1)
            binary_indices = pts_instance_mask.new(num_p, ).zero_()
            binary_indices[indices] = 1
            binary_indices = torch.nonzero(binary_indices & binary_pts_instance_mask, \
                                           as_tuple=False).squeeze(-1)
            binary_indices = binary_index[binary_indices]
            class_index = pts_semantic_mask[indices[0]]

            if class_index > background_flag:
                continue

            one_instance_feature = feature[:, binary_indices]
            context_features = one_instance_feature

            if context_features.shape[1] > 0:
                context_feature = torch.max(context_features, 1)[0].reshape(1, -1)

            if one_instance_feature.shape[1] < 1:
                continue

            one_instance_feature = torch.max(one_instance_feature, 1)[0].reshape(1, -1)

            class_name = class_names[class_index]

            if class_name not in new_dict.keys():
                new_dict[class_name] = [[], []]
            new_dict[class_name][0].append(one_instance_feature)
            if context_features.shape[1] > 0:
                new_dict[class_name][1].append(context_feature)

        batch_list.append(new_dict)

    return batch_list

def cls_gather(x, pts_semantic_masks, pts_instance_masks, points_xyzs, seed_indices, class_names=None, background_flag=17):

    points = x['fp_xyz'][-1]
    features = x['fp_features'][-1].detach() # (2*b, 128, 4096)
    # 注意，提取作为prototype时, 这里切断梯度了, 但想要计算cl_loss, 不应该切断梯度
    fp_indices = x['fp_indices'][-1]

    if class_names[0] == 'cabinet':
        num_p = 40000
    else:
        num_p = 20000

    batch_size = features.shape[0]
    batch_list = []
    for ii in range(batch_size):
        feature = features[ii] # (128,4096)
        pts_semantic_mask = pts_semantic_masks[ii] # 20000
        pts_instance_mask = pts_instance_masks[ii] # 20000
        # print('pts_semantic_mask: ', torch.unique(pts_semantic_mask)) # batch中的不同样本含有不同的semantic_mask

        new_dict = {}
        unique_pts_instance_mask = torch.unique(pts_instance_mask)
        # print('unique_pts_instance_mask: ', unique_pts_instance_mask)
        binary_pts_instance_mask = pts_instance_mask.new(num_p, ).zero_() # 20000 zeros
        binary_pts_instance_mask[fp_indices[ii]] = 1 # up to 1024 points set to 1

        index = binary_pts_instance_mask.new_ones(fp_indices.shape[-1]) # 4096 ones
        index = index.cumsum(dim=0) - 1 # [0~4095] (4096 idx)
        binary_index = pts_instance_mask.new(num_p, ).zero_()
        binary_index[fp_indices[ii]] = index # 记录点在grouped_features中的位置

        for i in unique_pts_instance_mask:
            indices = torch.nonzero(pts_instance_mask == i, as_tuple=False).squeeze(-1) # pts belonging to instance i
            binary_indices = pts_instance_mask.new(num_p, ).zero_()
            binary_indices[indices] = 1
            binary_indices = torch.nonzero(binary_indices & binary_pts_instance_mask, \
                                           as_tuple=False).squeeze(-1) # pts belonging to instance i && selected after vote
            binary_indices = binary_index[binary_indices]
            class_index = pts_semantic_mask[indices[0]] # get class of instance i

            # print('instance ', i, 'belongs to class ', class_index) ## instance -1 belongs to background

            if class_index > background_flag:
                continue

            one_instance_feature = feature[:, binary_indices] # one instance feature
            context_features = one_instance_feature

            if context_features.shape[1] > 0: # successfully extracted pts of instance i
                context_feature = torch.max(context_features, 1)[0].reshape(1, -1).detach() # (128,num_pts) is max pooled to (1,128)

            if one_instance_feature.shape[1] < 1: # extracted no point of instance i
                continue

            one_instance_feature = torch.max(one_instance_feature, 1)[0].reshape(1, -1) # (128,num_pts) is max pooled to (1,128)

            class_name = class_names[class_index]

            if class_name not in new_dict.keys():
                new_dict[class_name] = [[], []]
            new_dict[class_name][0].append(one_instance_feature.detach())
            if context_features.shape[1] > 0:
                new_dict[class_name][1].append(context_feature)

        batch_list.append(new_dict)

    return batch_list

def cls_prototype_grad(batch_list, context_compen, num=3, way=6): # 实际的num和way是多少?
    # context_compen = feature_norm(context_compen)
    # print('batch_list: ', batch_list[0].keys()) # 有的sample里根本就没有提取出待检测类别的点
    K_shot = num
    centroids = []
    batch_size = len(batch_list)
    # way = 4
    # print('way: ', way)
    one_prototype_dict = [[] for _ in range(batch_size)]
    all_batch_name = []
    for bs in range(batch_size):
        one_batch = batch_list[bs]
        other_batch = batch_list[0:bs] + batch_list[bs + 1:]
        one_batch_name = one_batch.keys()

        other_dict = {}
        for one_other_batch in other_batch:
            # print('other_batch: ', one_other_batch.keys())
            for name, features in one_other_batch.items():
                # print('features of %s : ' % name, len(features[0]))
                if name not in other_dict.keys():
                    other_dict[name] = [[], []]
                other_dict[name][0] += features[0]
                other_dict[name][1] += features[1]
                """
                append class names in one batch
                """
                if name not in all_batch_name:
                    all_batch_name.append(name)

        other_dict_center = {}
        for class_name in other_dict.keys():
            class_name_features = other_dict[class_name][0]
            class_name_contexts = other_dict[class_name][1]
            if len(class_name_features) >= K_shot:
                this_k_shot = K_shot
                class_name_features = class_name_features[:this_k_shot]
            else:
                this_k_shot = len(class_name_features)
                class_name_features = class_name_features*K_shot
                class_name_features = class_name_features[:K_shot]

            if len(class_name_contexts) >= K_shot:
                this_k_shot = K_shot
                class_name_contexts = class_name_contexts[:this_k_shot]
            else:
                this_k_shot = len(class_name_contexts)
                compen_num = K_shot - this_k_shot
                class_name_contexts += [context_compen] * compen_num

            instance_features = torch.cat(class_name_features, 0)
            this_center = torch.mean(instance_features, dim=0).reshape(1, -1)

            # this_center = feature_norm(this_center)
            instance_features = feature_norm(instance_features)
            this_center = feature_norm(this_center)

            # this_center = this_center.repeat(K_shot, 1)
            this_context = torch.cat(class_name_contexts, 0)

            if K_shot == 1:
                other_dict_center[class_name] = this_center
            else:
                other_dict_center[class_name] = torch.cat((this_center, instance_features), 0) # K+1个
            # other_dict_center[class_name] = this_center
            # other_dict_center[class_name] = torch.cat((this_center, instance_features, this_context), 0)

        # print('otherd_dict_center name: ', other_dict_center.keys())

        ###### other_dict_center.keys = desk, chair, dresser, bookshelf, dim:(k+1, channel), k|=1, 否则center就是样本的平均值

        N_way = way
        one_dict = {}
        one_dict_feature = []
        for name in one_batch_name: # 处理样本自身
            """
            append class names in one batch
            """
            if name not in all_batch_name:
                all_batch_name.append(name)

            if name in other_dict_center.keys():
                one_dict[name] = other_dict_center[name] # 加入的是其他sample中该类别的特征
                one_dict_feature.append(other_dict_center[name])
            else: # 如果其他sample中没有要检测的类别, 就用自己的特征
                if len(one_batch[name]) > K_shot:
                    feature = one_batch[name][0][:K_shot]
                else:
                    feature = one_batch[name][0]*K_shot
                    feature = feature[:K_shot]

                if len(one_batch[name][1]) >= K_shot:
                    single_context_feature = one_batch[name][1][:K_shot]
                else:
                    this_k_shot = len(one_batch[name][1])
                    single_context_feature = one_batch[name][1]
                    compen_num = K_shot - this_k_shot
                    single_context_feature += [context_compen] * compen_num

                instance_features = torch.cat(feature, 0)
                feature = torch.mean(instance_features, dim=0).reshape(1, -1)
                # feature = feature.repeat(K_shot, 1)
                # feature = feature_norm(feature)
                instance_features = feature_norm(instance_features)
                feature = feature_norm(feature)

                single_context_feature = torch.cat(single_context_feature, 0)

                if K_shot > 1:
                    feature = torch.cat((feature, instance_features), 0)

                # print(feature.shape)
                one_dict[name] = feature
                one_dict_feature.append(feature)
            # if len(one_dict) >= N_way:
            #     break

        for name in other_dict_center.keys(): # 自己有的类别+其他样本有的类别, 构成一个batch中所有的类别
            # print('name of %d in batch: ' % bs, name)

            # if len(one_dict) >= N_way: 会导致all_batch_name与one_dict.keys()不一致
            #     break
            if name not in one_dict.keys():
                one_dict[name] = other_dict_center[name]
                one_dict_feature.append(other_dict_center[name])

        # print('one_dict.keys(): ', one_dict.keys())
        # print('all_batch_name: ', all_batch_name)

        # TODO here
        for name in all_batch_name:
            one_prototype_dict[bs].append(one_dict[name][0].unsqueeze(0)) # 保证ls中的name顺序是一致的
        one_prototype_dict[bs] = torch.cat(one_prototype_dict[bs], dim=0).unsqueeze(0) # one_prototype_dict:[1,4,128]*batchsize

        # while len(one_dict_feature) < N_way:
        #     one_dict_feature.append(context_compen) # 3-shot的情况下，如果进行到了这一步，就会出现3*4+1=13的维度

        # one_dict_feature = torch.cat(one_dict_feature, 0).unsqueeze(0)
        # print('one_dict_feature: ', one_dict_feature.shape)

        # centroids.append(one_dict_feature)

    # centroids = torch.cat(centroids, 0)
    # print('one_dict: ', one_dict.keys()) # 每个样本顺序不同
    one_prototype_dict = torch.cat(one_prototype_dict, dim=0) # (16,4,128)

    return one_prototype_dict

def cls_prototype_grad_mf(batch_list, context_compen, num=3, way=6): # 实际的num和way是多少?
    # context_compen = feature_norm(context_compen)
    # print('batch_list: ', batch_list[0].keys()) # 有的sample里根本就没有提取出待检测类别的点
    K_shot = num
    centroids = []
    batch_size = len(batch_list)
    # way = 4
    # print('way: ', way)
    for bs in range(batch_size):
        one_batch = batch_list[bs]
        other_batch = batch_list[0:bs] + batch_list[bs + 1:]
        one_batch_name = one_batch.keys()

        other_dict = {}
        for one_other_batch in other_batch:
            # print('other_batch: ', one_other_batch.keys())
            for name, features in one_other_batch.items():
                # print('features of %s : ' % name, len(features[0]))
                if name not in other_dict.keys():
                    other_dict[name] = [[], []]
                other_dict[name][0] += features[0]
                other_dict[name][1] += features[1]

        other_dict_center = {}
        for class_name in other_dict.keys():
            class_name_features = other_dict[class_name][0]
            class_name_contexts = other_dict[class_name][1]
            if len(class_name_features) >= K_shot:
                this_k_shot = K_shot
                class_name_features = class_name_features[:this_k_shot]
            else:
                this_k_shot = len(class_name_features)
                class_name_features = class_name_features*K_shot
                class_name_features = class_name_features[:K_shot]

            if len(class_name_contexts) >= K_shot:
                this_k_shot = K_shot
                class_name_contexts = class_name_contexts[:this_k_shot]
            else:
                this_k_shot = len(class_name_contexts)
                compen_num = K_shot - this_k_shot
                class_name_contexts += [context_compen] * compen_num

            instance_features = torch.cat(class_name_features, 0)
            this_center = torch.mean(instance_features, dim=0).reshape(1, -1)

            # this_center = feature_norm(this_center)
            instance_features = feature_norm(instance_features)
            this_center = feature_norm(this_center)

            # this_center = this_center.repeat(K_shot, 1)
            this_context = torch.cat(class_name_contexts, 0)

            if K_shot == 1:
                other_dict_center[class_name] = this_center
            else:
                other_dict_center[class_name] = torch.cat((this_center, instance_features), 0) # K+1个
            # other_dict_center[class_name] = this_center
            # other_dict_center[class_name] = torch.cat((this_center, instance_features, this_context), 0)

        # print('otherd_dict_center name: ', other_dict_center.keys())

        ###### other_dict_center.keys = desk, chair, dresser, bookshelf, dim:(k+1, channel), k|=1, 否则center就是样本的平均值

        N_way = way
        one_dict = {}
        one_dict_feature = []
        for name in one_batch_name: # 处理样本自身
            if name in other_dict_center.keys():
                one_dict[name] = other_dict_center[name] # 加入的是其他sample中该类别的特征
                one_dict_feature.append(other_dict_center[name])
            else: # 如果其他sample中没有要检测的类别
                if len(one_batch[name]) > K_shot:
                    feature = one_batch[name][0][:K_shot]
                else:
                    feature = one_batch[name][0]*K_shot
                    feature = feature[:K_shot]

                if len(one_batch[name][1]) >= K_shot:
                    single_context_feature = one_batch[name][1][:K_shot]
                else:
                    this_k_shot = len(one_batch[name][1])
                    single_context_feature = one_batch[name][1]
                    compen_num = K_shot - this_k_shot
                    single_context_feature += [context_compen] * compen_num

                instance_features = torch.cat(feature, 0)
                feature = torch.mean(instance_features, dim=0).reshape(1, -1)
                # feature = feature.repeat(K_shot, 1)
                # feature = feature_norm(feature)
                instance_features = feature_norm(instance_features)
                feature = feature_norm(feature)

                single_context_feature = torch.cat(single_context_feature, 0)

                if K_shot > 1:
                    feature = torch.cat((feature, instance_features), 0)

                # print(feature.shape)
                one_dict[name] = feature
                one_dict_feature.append(feature)
            if len(one_dict) >= N_way:
                break

        for name in other_dict_center.keys():
            # print('name of %d in batch: ' % bs, name)
            # 每个batch里样本的name顺序是一致的, 是一次性检出4个类别的目标吗?
            if len(one_dict) >= N_way:
                break
            if name not in one_dict.keys():
                one_dict[name] = other_dict_center[name]
                one_dict_feature.append(other_dict_center[name])

        while len(one_dict_feature) < N_way:
            one_dict_feature.append(context_compen) # 3-shot的情况下，如果进行到了这一步，就会出现3*4+1=13的维度

        one_dict_feature = torch.cat(one_dict_feature, 0).unsqueeze(0)
        # print('one_dict_feature: ', one_dict_feature.shape)

        centroids.append(one_dict_feature)

    centroids = torch.cat(centroids, 0)
    # print('one_dict: ', one_dict.keys()) # 每个样本顺序不同

    return centroids

def cls_prototype(batch_list, context_compen, num=3, way=6): # 实际的num和way是多少?
    # context_compen = feature_norm(context_compen)
    # print('batch_list: ', batch_list[0].keys()) # 有的sample里根本就没有提取出待检测类别的点
    K_shot = num
    centroids = []
    batch_size = len(batch_list)
    # way = 4
    # print('way: ', way)
    for bs in range(batch_size):
        one_batch = batch_list[bs]
        other_batch = batch_list[0:bs] + batch_list[bs + 1:]
        one_batch_name = one_batch.keys()

        other_dict = {}
        for one_other_batch in other_batch:
            # print('other_batch: ', one_other_batch.keys())
            for name, features in one_other_batch.items():
                # print('features of %s : ' % name, len(features[0]))
                if name not in other_dict.keys():
                    other_dict[name] = [[], []]
                other_dict[name][0] += features[0]
                other_dict[name][1] += features[1]

        other_dict_center = {}
        for class_name in other_dict.keys():
            class_name_features = other_dict[class_name][0]
            class_name_contexts = other_dict[class_name][1]
            if len(class_name_features) >= K_shot:
                this_k_shot = K_shot
                class_name_features = class_name_features[:this_k_shot]
            else:
                this_k_shot = len(class_name_features)
                class_name_features = class_name_features*K_shot
                class_name_features = class_name_features[:K_shot]

            if len(class_name_contexts) >= K_shot:
                this_k_shot = K_shot
                class_name_contexts = class_name_contexts[:this_k_shot]
            else:
                this_k_shot = len(class_name_contexts)
                compen_num = K_shot - this_k_shot
                class_name_contexts += [context_compen] * compen_num

            instance_features = torch.cat(class_name_features, 0)
            this_center = torch.mean(instance_features, dim=0).reshape(1, -1)

            # this_center = feature_norm(this_center)
            instance_features = feature_norm(instance_features)
            this_center = feature_norm(this_center)

            # this_center = this_center.repeat(K_shot, 1)
            this_context = torch.cat(class_name_contexts, 0)

            if K_shot == 1:
                other_dict_center[class_name] = this_center
            else:
                other_dict_center[class_name] = torch.cat((this_center, instance_features), 0) # K+1个
            # other_dict_center[class_name] = this_center
            # other_dict_center[class_name] = torch.cat((this_center, instance_features, this_context), 0)

        # print('otherd_dict_center name: ', other_dict_center.keys())

        ###### other_dict_center.keys = desk, chair, dresser, bookshelf, dim:(k+1, channel), k|=1, 否则center就是样本的平均值

        N_way = way
        one_dict = {}
        one_dict_feature = []
        for name in one_batch_name: # 处理样本自身
            if name in other_dict_center.keys():
                one_dict[name] = other_dict_center[name] # 加入的是其他sample中该类别的特征
                one_dict_feature.append(other_dict_center[name])
            else: # 如果其他sample中没有要检测的类别
                if len(one_batch[name]) > K_shot:
                    feature = one_batch[name][0][:K_shot]
                else:
                    feature = one_batch[name][0]*K_shot
                    feature = feature[:K_shot]

                if len(one_batch[name][1]) >= K_shot:
                    single_context_feature = one_batch[name][1][:K_shot]
                else:
                    this_k_shot = len(one_batch[name][1])
                    single_context_feature = one_batch[name][1]
                    compen_num = K_shot - this_k_shot
                    single_context_feature += [context_compen] * compen_num

                instance_features = torch.cat(feature, 0)
                feature = torch.mean(instance_features, dim=0).reshape(1, -1)
                # feature = feature.repeat(K_shot, 1)
                # feature = feature_norm(feature)
                instance_features = feature_norm(instance_features)
                feature = feature_norm(feature)

                single_context_feature = torch.cat(single_context_feature, 0)

                if K_shot > 1:
                    feature = torch.cat((feature, instance_features), 0)

                # print(feature.shape)
                one_dict[name] = feature
                one_dict_feature.append(feature)
            if len(one_dict) >= N_way:
                break

        for name in other_dict_center.keys():
            # print('name of %d in batch: ' % bs, name)
            # 每个batch里样本的name顺序是一致的, 是一次性检出4个类别的目标吗?
            if len(one_dict) >= N_way:
                break
            if name not in one_dict.keys():
                one_dict[name] = other_dict_center[name]
                one_dict_feature.append(other_dict_center[name])

        while len(one_dict_feature) < N_way:
            one_dict_feature.append(context_compen) # 3-shot的情况下，如果进行到了这一步，就会出现3*4+1=13的维度

        one_dict_feature = torch.cat(one_dict_feature, 0).unsqueeze(0)
        # print('one_dict_feature: ', one_dict_feature.shape)

        centroids.append(one_dict_feature)

    centroids = torch.cat(centroids, 0)
    # print('one_dict: ', one_dict.keys()) # 每个样本顺序不同

    return centroids

def cls_prototype_support(batch_list, batch_size, num=3, way=6, compen_context=None, few_shot_class=None):

    K_shot = num
    other_batch = batch_list
    one_batch_name = few_shot_class

    other_dict = {}
    for one_other_batch in other_batch:
        for name, features in one_other_batch.items():
            if name not in other_dict.keys():
                other_dict[name] = [[], []]
            other_dict[name][0] += features[0]
            other_dict[name][1] += features[1]

    other_dict_center = {}
    for class_name in other_dict.keys():
        class_name_features = other_dict[class_name][0]
        class_name_contexts = other_dict[class_name][1]
        # print(len(class_name_features))
        if len(class_name_features) >= K_shot:
            this_k_shot = K_shot
            class_name_features = class_name_features[:this_k_shot]
        else:
            this_k_shot = len(class_name_features)
            class_name_features = class_name_features * K_shot
            class_name_features = class_name_features[:K_shot]

        if len(class_name_contexts) >= K_shot:
            this_k_shot = K_shot
            class_name_contexts = class_name_contexts[:this_k_shot]
        else:
            this_k_shot = len(class_name_contexts)
            compen_num = K_shot - this_k_shot
            class_name_contexts += [compen_context] * compen_num

        instance_features = torch.cat(class_name_features, 0)
        this_center = torch.mean(instance_features, dim=0).reshape(1, -1)

        instance_features = feature_norm(instance_features)
        this_center = feature_norm(this_center)

        this_context = torch.cat(class_name_contexts, 0)

        if K_shot == 1:
            other_dict_center[class_name] = this_center
        else:
            other_dict_center[class_name] = torch.cat((this_center, instance_features), 0)
        # other_dict_center[class_name] = this_center
        # other_dict_center[class_name] = torch.cat((this_center, instance_features, this_context), 0)

    N_way = way
    one_dict = {}
    one_dict_feature = []

    for name in one_batch_name:
        if name in other_dict_center.keys():

            one_dict[name] = other_dict_center[name]
            one_dict_feature.append(other_dict_center[name])

    one_dict_feature = torch.cat(one_dict_feature, 0).unsqueeze(0).repeat(batch_size, 1, 1)

    return one_dict_feature