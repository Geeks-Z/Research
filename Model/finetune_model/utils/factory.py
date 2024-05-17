def get_model(model_name, args):
    name = model_name.lower()
    if name == "vit":
        from models.vit import Learner

    # elif name == "finetune":
    #     from models.finetune import Learner

    else:
        assert 0
    
    return Learner(args)
