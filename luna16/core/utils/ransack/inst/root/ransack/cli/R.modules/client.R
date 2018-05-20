import::here("%>%", .from = "magrittr")
import::here(arg_parser, add_argument, parse_args, .from = "argparser")

modules::export("argv")

argv <- arg_parser("ransack") %>%
  add_argument("<command>",
    help = "where <command> is one of: hoist, clean"
  ) %>%
  parse_args
