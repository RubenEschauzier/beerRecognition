def load_mobilenet_ssd(num_classes, check_point, pipeline_config):
    configs = config_util.get_configs_from_pipeline_file(pipeline_config)
    model_config = configs['model']
    model_config.ssd.num_classes = num_classes
    model_config.ssd.freeze_batchnorm = True
    detection_model = model_builder.build(
        model_config=model_config, is_training=True)
    # print(tf.train.list_variables(tf.train.latest_checkpoint(check_point)))
    # print(detection_model._box_predictor._prediction_heads)

    fake_box_predictor = tf.compat.v2.train.Checkpoint(
        _prediction_heads=detection_model._box_predictor._prediction_heads['box_encodings'],
        # _prediction_heads=detection_model._box_predictor._prediction_heads,
        #    (i.e., the classification head that we *will not* restore)
        # _box_prediction_head=detection_model._box_predictor._box_prediction_head,
    )
    fake_model = tf.compat.v2.train.Checkpoint(
        _feature_extractor=detection_model._feature_extractor,
        _box_predictor=fake_box_predictor)
    ckpt = tf.compat.v2.train.Checkpoint(model=fake_model)
    ckpt.restore(os.path.join(checkpoint_path, 'ckpt-0')).expect_partial()

    # Run model through a dummy image so that variables are created
    image, shapes = detection_model.preprocess(tf.zeros([1, 300, 300, 3]))
    prediction_dict = detection_model.predict(image, shapes)
    _ = detection_model.postprocess(prediction_dict, shapes)
    print('Weights restored!')
    print(detection_model)
    return detection_model


def train_model(train_images_list, train_bound_boxes_list, train_classes_list, val_images, val_bound_boxes, val_classes, model, epochs,
                size_input, lr, batch_size):
    """
    :param train_images: Not randomly sampled yet
    :param train_bound_boxes:
    :param train_classes:
    :return:
    """
    val_images_tensor = [tf.expand_dims(tf.convert_to_tensor(val_images[x], dtype=tf.float32), 0)
                         for x in range(len(val_images))]
    val_bound_boxes_tensor = [tf.convert_to_tensor(val_bound_boxes[x], dtype=tf.float32)
                              for x in range(len(val_bound_boxes))]
    val_classes_tensor = [tf.convert_to_tensor(val_classes[x], dtype=tf.float32)
                          for x in range(len(val_classes))]

    trainable_variables = model.trainable_variables
    to_fine_tune = trainable_variables

    shapes = tf.constant(batch_size * [[300, 300, 3]], dtype=tf.int32)
    optimizer = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0.9)

    train_step_function = perform_step_training(model, optimizer, to_fine_tune, batch_size)
    validation_step_function = perform_step_validation(model, batch_size)

    num_batches = math.ceil(len(train_images_list) / batch_size)
    num_val_batches = math.ceil(len(val_images) / batch_size)

    ckpt = tf.train.Checkpoint(model=model)
    manager = tf.train.CheckpointManager(ckpt, './fine_tuned_model', max_to_keep=1)

    for epoch in range(epochs):
        epoch_train_loss = 0
        epoch_val_loss = 0

        images, bounding_boxes = process_images_sample_randomly(train_images_list, train_bound_boxes_list, size_input)
        train_images = [tf.expand_dims(tf.convert_to_tensor(images[x], dtype=tf.float32), 0)
                        for x in range(len(images))]
        train_boxes = [tf.convert_to_tensor(bounding_boxes[x], dtype=tf.float32)
                       for x in range(len(bounding_boxes))]
        train_classes = [tf.convert_to_tensor(train_classes_list[x], dtype=tf.float32)
                         for x in range(len(train_classes_list))]

        # Shuffle data before our epoch
        all_data = list(zip(train_images, train_boxes, train_classes))
        random.shuffle(all_data)
        train_images, train_boxes, train_classes = zip(*all_data)

        for batch in range(num_batches):
            if batch == num_batches - 1:
                batch_images = train_images[batch * batch_size:]
                batch_boxes = train_boxes[batch * batch_size:]
                batch_classes = train_classes[batch * batch_size:]
            else:
                batch_images = train_images[batch * batch_size:(batch + 1) * batch_size]
                batch_boxes = train_boxes[batch * batch_size:(batch + 1) * batch_size]
                batch_classes = train_classes[batch * batch_size:(batch + 1) * batch_size]

            total_loss = train_step_function(batch_images, batch_boxes, batch_classes)
            epoch_train_loss += total_loss
            # if batch % 10 == 0:
            #     print('INFO: Batch {}/{}, Loss: {}'.format(batch, num_batches, total_loss.numpy()/batch_size),
            #           flush=True)

        for batch in range(num_val_batches):
            if batch == num_batches - 1:
                batch_images = val_images_tensor[batch * batch_size:]
                batch_boxes = val_bound_boxes_tensor[batch * batch_size:]
                batch_classes = val_classes_tensor[batch * batch_size:]
            else:
                batch_images = val_images_tensor[batch * batch_size:(batch + 1) * batch_size]
                batch_boxes = val_bound_boxes_tensor[batch * batch_size:(batch + 1) * batch_size]
                batch_classes = val_classes_tensor[batch * batch_size:(batch + 1) * batch_size]

            total_loss = validation_step_function(batch_images, batch_boxes, batch_classes)
            epoch_val_loss += total_loss

        print('INFO: Epoch: {}/{}, Av. Train Loss: {}, Av. Val Loss: {}'.
              format(epoch + 1, epochs, epoch_train_loss / len(train_images), epoch_val_loss / len(val_images)))
        manager.save()

    return model


def perform_step_training(model, optimizer, to_fine_tune, batch_size):
    def train_step(image_tensors, bounding_box_tensors, class_tensors):
        shapes = tf.constant(batch_size * [[300, 300, 3]], dtype=tf.int32)
        model.provide_groundtruth(groundtruth_boxes_list=bounding_box_tensors, groundtruth_classes_list=class_tensors)
        with tf.GradientTape() as tape:
            preprocessed_images = tf.concat([model.preprocess(image_tensor)[0] for image_tensor in image_tensors],
                                            axis=0)
            prediction_dict = model.predict(preprocessed_images, shapes)
            losses_dict = model.loss(prediction_dict, shapes)
            total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
            gradients = tape.gradient(total_loss, to_fine_tune)
            optimizer.apply_gradients(zip(gradients, to_fine_tune))
        return total_loss

    return train_step


def perform_step_validation(model, batch_size):
    def validation_step(val_image_tensors, val_bounding_box_tensors, val_class_tensors):
        shapes = tf.constant(batch_size * [[300, 300, 3]], dtype=tf.int32)
        model.provide_groundtruth(groundtruth_boxes_list=val_bounding_box_tensors,
                                  groundtruth_classes_list=val_class_tensors)
        preprocessed_images = tf.concat([model.preprocess(image_tensor)[0] for image_tensor in val_image_tensors],
                                        axis=0)
        prediction_dict = model.predict(preprocessed_images, shapes)
        losses_dict = model.loss(prediction_dict, shapes)
        total_loss = losses_dict['Loss/localization_loss'] + losses_dict['Loss/classification_loss']
        return total_loss

    return validation_step

# pipeline_config = 'ssd_mobilenet/pipeline.config'
# checkpoint_path = 'ssd_mobilenet/checkpoint'
#
# ssd_model = load_mobilenet_ssd(2, checkpoint_path, pipeline_config)
# print('INFO: Fine-tuning model')
# fine_tuned_model = train_model(scaled_training_images, scaled_training_annotations, training_classes,
#                                scaled_val_images, scaled_val_annotations, validation_classes,
#                                ssd_model, 1, 224, 5e-4, 8)
