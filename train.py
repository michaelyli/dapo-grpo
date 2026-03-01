"""Minimal GRPO training on deduplicated DAPO-17k with Qwen3."""

import argparse

from datasets import load_dataset
from transformers import AutoModelForCausalLM
from trl import GRPOConfig, GRPOTrainer

from rewards import accuracy_reward

BASIC_SOLVE_PROMPT = """
Question:
{question}

---
Please reason step by step to solve the question above and put final answer within \\boxed{{}}.
""".strip()


def build_prompt(example):
    """Build a plain-text prompt for a base model."""
    example["prompt"] = BASIC_SOLVE_PROMPT.format(question=example["prompt"])
    return example


def load_train_dataset():
    ds = load_dataset("ftajwar/deduplicated_dapo_dataset", split="train")
    ds = ds.rename_column("answer", "solution")
    ds = ds.map(build_prompt)
    return ds


def load_eval_dataset():
    ds = load_dataset("HuggingFaceH4/MATH-500", split="test")
    ds = ds.rename_column("problem", "prompt")
    ds = ds.remove_columns("solution")
    ds = ds.rename_column("answer", "solution")
    ds = ds.map(build_prompt)
    return ds


def parse_args():
    parser = argparse.ArgumentParser(description="GRPO training on DAPO-17k")
    parser.add_argument("--model_name_or_path", type=str, default="Qwen/Qwen3-1.7B-Base")
    parser.add_argument("--output_dir", type=str, default="./output")
    parser.add_argument("--max_steps", type=int, default=-1)
    parser.add_argument("--num_train_epochs", type=int, default=1)
    parser.add_argument("--learning_rate", type=float, default=1e-6)
    parser.add_argument("--num_generations", type=int, default=8)
    parser.add_argument("--per_device_train_batch_size", type=int, default=1)
    parser.add_argument("--per_device_eval_batch_size", type=int, default=8)
    parser.add_argument("--gradient_accumulation_steps", type=int, default=4)
    parser.add_argument("--max_completion_length", type=int, default=1360)
    parser.add_argument("--max_prompt_length", type=int, default=256)
    parser.add_argument("--loss_type", type=str, default="dr_grpo")
    parser.add_argument("--beta", type=float, default=0.0)
    parser.add_argument("--temperature", type=float, default=1.0)
    parser.add_argument("--num_iterations", type=int, default=1)
    parser.add_argument("--max_grad_norm", type=float, default=1.0)
    parser.add_argument("--adam_epsilon", type=float, default=1e-15)
    parser.add_argument("--weight_decay", type=float, default=1e-4)
    parser.add_argument("--adam_beta1", type=float, default=0.9)
    parser.add_argument("--adam_beta2", type=float, default=0.95)
    parser.add_argument("--save_steps", type=int, default=100)
    parser.add_argument("--eval_steps", type=int, default=100)
    parser.add_argument("--logging_steps", type=int, default=1)
    parser.add_argument("--bf16", action="store_true", default=True)
    parser.add_argument("--run_name", type=str, default=None)
    return parser.parse_args()


def main():
    args = parse_args()

    train_dataset = load_train_dataset()
    eval_dataset = load_eval_dataset()

    training_args = GRPOConfig(
        output_dir=args.output_dir,
        max_steps=args.max_steps,
        num_train_epochs=args.num_train_epochs,
        learning_rate=args.learning_rate,
        num_generations=args.num_generations,
        per_device_train_batch_size=args.per_device_train_batch_size,
        per_device_eval_batch_size=args.per_device_eval_batch_size,
        gradient_accumulation_steps=args.gradient_accumulation_steps,
        max_completion_length=args.max_completion_length,
        max_prompt_length=args.max_prompt_length,
        loss_type=args.loss_type,
        beta=args.beta,
        temperature=args.temperature,
        num_iterations=args.num_iterations,
        max_grad_norm=args.max_grad_norm,
        adam_epsilon=args.adam_epsilon,
        weight_decay=args.weight_decay,
        adam_beta1=args.adam_beta1,
        adam_beta2=args.adam_beta2,
        save_steps=args.save_steps,
        eval_steps=args.eval_steps,
        eval_strategy="steps",
        logging_steps=args.logging_steps,
        bf16=False,
        report_to="wandb",
        run_name=args.run_name,
        log_completions=True,
        log_on_each_node=False,
        lr_scheduler_type="constant",
    )

    model = AutoModelForCausalLM.from_pretrained(
        args.model_name_or_path,
        torch_dtype="auto",
    )

    trainer = GRPOTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=eval_dataset,
        reward_funcs=[accuracy_reward],
    )

    trainer.train()
    trainer.save_model(args.output_dir)


if __name__ == "__main__":
    main()
