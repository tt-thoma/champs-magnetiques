from optparse import OptionParser, Values

from ._result import GitHubTestResult

parser: OptionParser = OptionParser()
parser.add_option("-f", "--full", action="store_true", dest="full", default=False)
parser.add_option(
    "-g",
    "--github",
    action="store_const",
    dest="resultclass",
    default=None,
    const=GitHubTestResult,
)
parser.add_option(
    "-e", "--examples", action="store_true", dest="examples", default=False
)
parser.add_option(
    "-l", "--local", action="store_true", dest="local", default=False
)
opts: Values
args: list[str]
opts, args = parser.parse_args()
