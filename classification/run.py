import arguments
import main


if __name__ == '__main__':
    args = arguments.parse_args()
    logger, model = main.run(args, log_interval=100)
