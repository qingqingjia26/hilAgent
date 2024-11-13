import argparse


def agent_parser(input: str):
    # 创建 ArgumentParser 对象
    parser = argparse.ArgumentParser(
        description="Process some commands.", add_help=False
    )
    parser.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help message",
    )
    # 添加子命令解析器
    subparsers = parser.add_subparsers(help="sub-command help", dest="command")

    # 创建子命令 no
    parser_no = subparsers.add_parser(
        "no", help="disagree with the result and put a response", add_help=False
    )
    parser_no.add_argument(
        "input",
        type=str,
        nargs="*",
        help="Customer's response required for the result",
    )
    parser_no.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help message",
    )

    # 创建子命令 'y'
    parser_yes = subparsers.add_parser(
        "y", help="yes command. to continue", add_help=False
    )

    parser_yes.add_argument(
        "-n",
        "--n",
        type=int,
        default=0,
        help="Number of Instances to Skip Customer Inquiry",
    )
    parser_yes.add_argument(
        "-h",
        "--help",
        action="store_true",
        help="Show this help message",
    )

    # 创建子命令exit
    parser_exit = subparsers.add_parser(
        "exit", help="exit command", add_help=False
    )

    subparser_dict = {
        "no": parser_no,
        "y": parser_yes,
        "exit": parser_exit,
    }
    # 将字符串分割成列表，模拟命令行参数
    args_list = input.split()
    # 解析参数
    try:
        args = parser.parse_args(args_list)
    except SystemExit:
        help_info = parser.format_help()
        return None, help_info
    if args.command is None:
        help_info = parser.format_help()
        return args, help_info
    if args.command == "exit":
        print("exit the program as requested")
        exit(0)
    if args.help:
        help_info = subparser_dict[args.command].format_help()
        return args, help_info
    if args.command == "no" and args.input == []:
        help_info = subparser_dict["no"].format_help()
        return args, help_info
    return args, ""
