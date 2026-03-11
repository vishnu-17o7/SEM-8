from uuid import uuid4

from sqlalchemy.orm import Session

from app.ingestion.base import BaseParser
from app.models.entities import TransactionRaw
from app.models.enums import ScenarioType


class IngestionService:
    def ingest_file(
        self,
        db: Session,
        parser: BaseParser,
        file_path: str,
        scenario_type: ScenarioType,
    ) -> str:
        batch_id = str(uuid4())
        parsed = parser.parse(file_path)
        for rec in parsed:
            db.add(
                TransactionRaw(
                    ingestion_batch_id=batch_id,
                    source_type=parser.source_type,
                    source_system=parser.source_system,
                    scenario_type=scenario_type,
                    file_name=file_path.split("/")[-1],
                    row_number=rec.row_number,
                    raw_payload=rec.payload,
                    parser_name=parser.__class__.__name__,
                )
            )
        db.commit()
        return batch_id
