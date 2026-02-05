#!/bin/bash
#
# fal-scaffold.sh - Fetch fal.ai model info and generate scaffold files
#
# Usage:
#   ./fal-scaffold.sh <endpoint-id> [app-dir]
#   ./fal-scaffold.sh fal-ai/dia-tts dia-tts
#   ./fal-scaffold.sh fal-ai/reve/text-to-image reve
#
# If app-dir exists, generates MODEL.md and PRICING.md there.
# Otherwise generates in current directory.
#
# Requires:
#   - FAL_KEY environment variable (or .fal.key file)
#   - curl, python3

set -e

ENDPOINT_ID="${1:-}"
APP_DIR="${2:-}"

if [[ -z "$ENDPOINT_ID" ]]; then
    echo "Usage: $0 <endpoint-id> [app-dir]"
    echo "Example: $0 fal-ai/dia-tts dia-tts"
    exit 1
fi

# Load FAL_KEY
if [[ -z "$FAL_KEY" ]]; then
    SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
    if [[ -f "$SCRIPT_DIR/.fal.key" ]]; then
        FAL_KEY=$(cat "$SCRIPT_DIR/.fal.key")
    elif [[ -f "../.fal.key" ]]; then
        FAL_KEY=$(cat "../.fal.key")
    else
        echo "Error: FAL_KEY not set and .fal.key not found"
        exit 1
    fi
fi

# Determine output directory
if [[ -n "$APP_DIR" && -d "$APP_DIR" ]]; then
    OUTPUT_DIR="$APP_DIR"
elif [[ -n "$APP_DIR" ]]; then
    echo "Warning: $APP_DIR does not exist, using current directory"
    OUTPUT_DIR="."
else
    OUTPUT_DIR="."
fi

# Extract model family for related search (e.g., "dia" from "fal-ai/dia-tts")
MODEL_FAMILY=$(echo "$ENDPOINT_ID" | sed 's|fal-ai/||' | cut -d'/' -f1 | cut -d'-' -f1)

echo "=== fal.ai Model Scaffold ==="
echo "Endpoint: $ENDPOINT_ID"
echo "Model family: $MODEL_FAMILY"
echo "Output dir: $OUTPUT_DIR"
echo ""

# Step 0: Search for related endpoints
echo "=== Step 0: Searching for related endpoints (q=$MODEL_FAMILY) ==="
RELATED=$(curl -s "https://api.fal.ai/v1/models?q=$MODEL_FAMILY&limit=20")

echo "$RELATED" | python3 -c "
import sys, json
data = json.load(sys.stdin)
models = data.get('models', [])
if models:
    print(f'Found {len(models)} related endpoint(s):')
    for m in models:
        eid = m.get('endpoint_id', '')
        cat = m.get('metadata', {}).get('category', '')
        name = m.get('metadata', {}).get('display_name', '')
        status = m.get('metadata', {}).get('status', '')
        print(f'  - {eid} ({cat}) - {name} [{status}]')
else:
    print('No related endpoints found.')
"
echo ""

# Step 1: Fetch OpenAPI schema
echo "=== Step 1: Fetching OpenAPI schema ==="
SCHEMA=$(curl -s "https://api.fal.ai/v1/models?endpoint_id=$ENDPOINT_ID&expand=openapi-3.0")

# Check if model found
MODEL_COUNT=$(echo "$SCHEMA" | python3 -c "import sys,json; print(len(json.load(sys.stdin).get('models',[])))")
if [[ "$MODEL_COUNT" == "0" ]]; then
    echo "Error: Model not found: $ENDPOINT_ID"
    exit 1
fi

echo "Schema fetched successfully."
echo ""

# Step 2: Fetch pricing
echo "=== Step 2: Fetching pricing ==="
PRICING=$(curl -s -H "Authorization: Key $FAL_KEY" "https://api.fal.ai/v1/models/pricing?endpoint_id=$ENDPOINT_ID")
echo "Pricing fetched successfully."
echo ""

# Step 3: Generate MODEL.md
echo "=== Step 3: Generating MODEL.md ==="

echo "$SCHEMA" | python3 -c "
import sys, json

data = json.load(sys.stdin)
model = data['models'][0]
meta = model.get('metadata', {})
openapi = model.get('openapi', {})
schemas = openapi.get('components', {}).get('schemas', {})

endpoint_id = model.get('endpoint_id', '')
display_name = meta.get('display_name', '')
category = meta.get('category', '')
description = meta.get('description', '')

# Find input/output schemas
input_schema = None
output_schema = None
for name, schema in schemas.items():
    if name.endswith('Input') and name != 'BaseInput':
        input_schema = (name, schema)
    if name.endswith('Output') and name != 'BaseOutput':
        output_schema = (name, schema)

print(f'# Model: {endpoint_id}')
print()
print('## Endpoint')
print(f'\`{endpoint_id}\`')
print()
print('## Category')
print(category)
print()
print('## Description')
print(description)
print()

if input_schema:
    name, schema = input_schema
    props = schema.get('properties', {})
    required = schema.get('required', [])
    order = schema.get('x-fal-order-properties', list(props.keys()))

    print('## Input Schema')
    print()

    req_fields = [f for f in order if f in required]
    opt_fields = [f for f in order if f not in required]

    if req_fields:
        print('### Required Fields')
        for field in req_fields:
            prop = props.get(field, {})
            ftype = prop.get('type', prop.get('allOf', [{}])[0].get('\$ref', 'unknown').split('/')[-1])
            desc = prop.get('description', '')
            examples = prop.get('examples', [])
            example_str = f' (example: {examples[0]})' if examples else ''
            print(f'- \`{field}\` ({ftype}): {desc}{example_str}')
        print()

    if opt_fields:
        print('### Optional Fields')
        for field in opt_fields:
            prop = props.get(field, {})
            ftype = prop.get('type', 'unknown')
            desc = prop.get('description', '')
            default = prop.get('default', None)
            default_str = f', default: {default}' if default is not None else ''
            print(f'- \`{field}\` ({ftype}{default_str}): {desc}')
        print()

if output_schema:
    name, schema = output_schema
    props = schema.get('properties', {})
    order = schema.get('x-fal-order-properties', list(props.keys()))

    print('## Output Schema')
    for field in order:
        prop = props.get(field, {})
        ftype = prop.get('type', prop.get('allOf', [{}])[0].get('\$ref', 'unknown').split('/')[-1])
        desc = prop.get('description', '')
        print(f'- \`{field}\` ({ftype}): {desc}')
    print()

print('## Notes')
print('- [Add implementation notes here]')
" > "$OUTPUT_DIR/MODEL.md"

echo "Created $OUTPUT_DIR/MODEL.md"
echo ""

# Step 4: Generate PRICING.md
echo "=== Step 4: Generating PRICING.md ==="

echo "$PRICING" | python3 -c "
import sys, json

data = json.load(sys.stdin)
prices = data.get('prices', [])

if not prices:
    print('# Pricing')
    print()
    print('No pricing data found for this endpoint.')
    sys.exit(0)

price = prices[0]
endpoint_id = price.get('endpoint_id', '')
unit_price = price.get('unit_price', 0)
unit = price.get('unit', 'unknown')
currency = price.get('currency', 'USD')

# Calculate microcents
microcents = int(unit_price * 100_000_000)

# Determine price variable name and CEL expression
unit_map = {
    'image': ('per_megapixel', '(double(outputs[0].width) * double(outputs[0].height) / 1000000.0) * double(prices.per_megapixel)'),
    'second': ('per_second', 'double(outputs[0].seconds) * double(prices.per_second)'),
    '1000 characters': ('per_1k_characters', '(double(size(inputs[0].text)) / 1000.0) * double(prices.per_1k_characters)'),
    'minute': ('per_minute', '(double(outputs[0].seconds) / 60.0) * double(prices.per_minute)'),
    'request': ('per_run', 'double(prices.per_run)'),
}

var_name, cel_expr = unit_map.get(unit, ('per_unit', 'double(prices.per_unit)'))

print(f'# Pricing: {endpoint_id.split(\"/\")[-1]}')
print()
print('## fal.ai Base Price')
print(f'- Endpoint: \`{endpoint_id}\`')
print(f'- Price: \${unit_price} per {unit}')
print(f'- Currency: {currency}')
print()
print('## Price Variables (microcents)')
print(f'- \`{var_name}\`: {microcents} (= \${unit_price} * 100000000)')
print()
print('## CEL Expressions')
print()
print('### inference_expression')
print('\`\`\`cel')
print(cel_expr)
print('\`\`\`')
print()
print('### pricing_description')
print('\`\`\`cel')
print(f'\"\${unit_price} per {unit}\"')
print('\`\`\`')
print()
print('## Calculation Notes')
print(f'fal.ai charges \${unit_price} per {unit}.')
" > "$OUTPUT_DIR/PRICING.md"

echo "Created $OUTPUT_DIR/PRICING.md"
echo ""

echo "=== Done ==="
echo ""
echo "Files generated in: $OUTPUT_DIR"
if [[ "$OUTPUT_DIR" == "." ]]; then
    echo ""
    echo "Next steps:"
    echo "  1. Review related endpoints above - consolidate if needed"
    echo "  2. Run: infsh app init <app-name>"
    echo "  3. Move files: mv MODEL.md PRICING.md <app-dir>/"
    echo "  4. Copy helper: cp fal_helper.py <app-dir>/"
    echo "  5. Implement inference.py"
fi
