import timm
model = timm.create_model("vit_base_patch16_224_in21k",pretrained=True, num_classes=0)
print(model)

# from peft import LoraConfig, get_peft_model
#
# config = LoraConfig(
#     r=16,
#     lora_alpha=16,
#     target_modules=["query", "value"], # 定义需要做低秩适配的部分
#     lora_dropout=0.1,
#     bias="none",
#     modules_to_save=["classifier"],
# )
# lora_model = get_peft_model(model, config)
# print_trainable_parameters(lora_model)