magics::.__file__(function(x) {
  x
})

import::here("%>%",
  .from = "magrittr")

modules::export("argv")

argv <- argparser::arg_parser("ransack") %>%
  argparser::add_argument("--e",
    short = "-e",
    help = "path argument",
    nargs = Inf
  ) %>%
  argparser::add_argument("<command>",
    help = "where <command> is one of: hoist, clean, cranify, packify, snapshot"
  ) %>%
  argparser::parse_args()
