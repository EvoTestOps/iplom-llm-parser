DEFAULT_PROMPT = (
    "Given log messages below, extract a single common template.\n"
    "Replace each dynamic value with a typed placeholder based on its category.\n"
    "Keep static parts, punctuation and whitespace exactly as-is. There might be no variables in the logs.\n\n"
    "# Placeholder types:\n"
    "  <OID> - Session IDs, user IDs, etc.\n"
    "  <LOI> - Paths, URIs, IP addresses\n"
    "  <OBN> - Object names, task names, job names\n"
    "  <TID> - Type indicators\n"
    "  <SID> - Numerical switch/flag indicators\n"
    "  <TDA> - Timestamps, durations\n"
    "  <CRS> - Memory, disk space, byte counts\n"
    "  <OBA> - Counts of objects (errors, nodes, etc.)\n"
    "  <STC> - Numerical error/status codes\n"
    "  <OTP> - Any other dynamic value\n\n"
    "# Example:\n"
    "  Logs:\n"
    "    `Connecting to 192.168.1.1:9000 as user admin`\n"
    "    `Connecting to 10.0.0.5:8080 as user root`\n"
    "  Template: `Connecting to <LOI> as user <OID>`\n\n"
    "Logs:\n"
    "{}\n\n"
    "Template:"
)

NO_EXAMPLE_PROMPT = (
    "Given log messages below, extract a single common template.\n"
    "Replace each dynamic value with a typed placeholder based on its category.\n"
    "Keep static parts, punctuation and whitespace exactly as-is. There might be no variables in the logs.\n\n"
    "# Placeholder types:\n"
    "  <OID> - Session IDs, user IDs, etc.\n"
    "  <LOI> - Paths, URIs, IP addresses\n"
    "  <OBN> - Object names, task names, job names\n"
    "  <TID> - Type indicators\n"
    "  <SID> - Numerical switch/flag indicators\n"
    "  <TDA> - Timestamps, durations\n"
    "  <CRS> - Memory, disk space, byte counts\n"
    "  <OBA> - Counts of objects (errors, nodes, etc.)\n"
    "  <STC> - Numerical error/status codes\n"
    "  <OTP> - Any other dynamic value\n\n"
    "Logs:\n"
    "{}\n\n"
    "Template:"
)

SIMPLE_PROMPT = (
    "Given log messages below, extract a single common template.\n"
    "Replace each dynamic variable with <*>.\n"
    "Keep static parts, punctuation and whitespace exactly as-is. There might be no variables in the logs.\n\n"
    "Logs:\n"
    "{}\n\n"
    "Template:"
)

PROMPTS: dict[str, str] = {
    "default": DEFAULT_PROMPT,
    "no_example": NO_EXAMPLE_PROMPT,
    "simple": SIMPLE_PROMPT,
}
