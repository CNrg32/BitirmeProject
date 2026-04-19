# AWS DynamoDB Transcript Storage

This integration stores `data/labels/auto_transcripts.csv` records in DynamoDB.

## Table

Recommended table:

- Table name: `EmergencyAutoTranscripts`
- Partition key: `transcript_id` (String)
- Billing mode: on-demand

Create with AWS CLI:

```bash
aws dynamodb create-table \
  --table-name EmergencyAutoTranscripts \
  --attribute-definitions AttributeName=transcript_id,AttributeType=S \
  --key-schema AttributeName=transcript_id,KeyType=HASH \
  --billing-mode PAY_PER_REQUEST \
  --region eu-central-1
```

## Environment

Set these for the backend or import script:

```bash
TRANSCRIPTS_DYNAMODB_TABLE=EmergencyAutoTranscripts
AWS_REGION=eu-central-1
```

Credentials should come from the normal AWS provider chain:

- `aws configure`
- IAM role on EC2/ECS/Lambda
- environment variables such as `AWS_ACCESS_KEY_ID` and `AWS_SECRET_ACCESS_KEY`

## Import CSV

From the project root:

```bash
PYTHONPATH=src python scripts/import_auto_transcripts_to_dynamodb.py
```

Custom table/region:

```bash
PYTHONPATH=src python scripts/import_auto_transcripts_to_dynamodb.py \
  --table EmergencyAutoTranscripts \
  --region eu-central-1
```

## API

After the backend is running:

- `POST /transcripts/import-auto-csv`
- `GET /transcripts?limit=50`
- `GET /transcripts/{transcript_id}`

If AWS configuration is missing, the API returns `503` with the missing setup detail.
