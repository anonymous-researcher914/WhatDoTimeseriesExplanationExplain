from experiments.run_beijing_multitask import run_beijing
from experiments.run_var_omic import run_var



def main():
    print("Running Synthetic Experiments")
    run_var()

    print("-----------------------------------------------")
    print("Running Case Studies on Beijing Dataset")
    run_beijing()


if __name__ == "__main__":
    main()
