import argparse
import torch
import yaml
from luxonis_train.core import Trainer

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("-cfg", "--config", type=str, required=True, help="Configuration file to use")
    parser.add_argument("--accelerator", type=str, help="Accelerator to use ('gpu' or 'cpu')", 
        default="gpu" if torch.cuda.is_available() else "cpu")
    parser.add_argument("--devices", default=None, nargs="+", help="Devices to use (e.g. 1 2)")
    args = parser.parse_args()
   
    if args.devices: # convert list of strings to ints
        args.devices = [int(x) for x in args.devices]

    with open(args.config) as f:
        cfg = yaml.load(f, Loader=yaml.SafeLoader)

    args_dict = vars(args)

    trainer = Trainer(args_dict, cfg)
    trainer.run()
    
    trainer.test()

    # Example: run in new thread
    # import time
    # trainer.run(new_thread=True)
    # while True:
    #     time.sleep(5)
    #     print(trainer.get_status())
