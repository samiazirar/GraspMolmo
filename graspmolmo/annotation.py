from pydantic import BaseModel
from enum import Enum

class GraspLabel(str, Enum):
    GOOD = "good"
    BAD = "bad"
    INFEASIBLE = "infeasible"

class JudgementLabel(str, Enum):
    ACCURATE = "accurate"
    INACCURATE = "inaccurate"
    UNCERTAIN = "uncertain"

class Object(BaseModel, frozen=True):
    object_category: str
    object_id: str

class Annotation(BaseModel, frozen=True):
    obj: Object
    grasp_id: int
    obj_description: str
    grasp_description: str
    grasp_label: GraspLabel
    is_mesh_malformed: bool = False
    user_id: str = ""
    time_taken: float = -1.0
    study_id: str = ""

class Judgement(BaseModel, frozen=True):
    annotation: Annotation
    judgement_label: JudgementLabel
    user_id: str = ""
    study_id: str = ""
    time_taken: float = -1.0
    correct_grasp_description: str | None = None

class QuestionResult(BaseModel, frozen=True):
    question_idx: int
    correct: bool
    time_taken: float  # in seconds

class PracticeResult(BaseModel, frozen=True):
    user_id: str
    total_time: float  # in seconds
    question_results: list[QuestionResult]
    timestamp: str  # in mmdd_hhmm format
    study_id: str = ""
