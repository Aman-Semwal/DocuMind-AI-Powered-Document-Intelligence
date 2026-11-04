"""Custom exceptions, each mapped to one HTTP status code (roadmap section 9, phase 4)."""

from __future__ import annotations


class AppError(Exception):
    status_code = 500
    code = "internal_error"
    default_message = "Something went wrong on the server."

    def __init__(self, message: str | None = None) -> None:
        self.message = message or self.default_message
        super().__init__(self.message)


class BadRequest(AppError):
    status_code = 400
    code = "bad_request"
    default_message = "The request is not valid."


class Unauthorized(AppError):
    status_code = 401
    code = "unauthorized"
    default_message = "Missing or invalid API key."


class NotFound(AppError):
    status_code = 404
    code = "not_found"
    default_message = "The requested item was not found."


class DuplicateDocument(AppError):
    status_code = 409
    code = "duplicate_document"
    default_message = "This document has already been uploaded."

    def __init__(self, message: str | None = None, existing_id: str | None = None) -> None:
        super().__init__(message)
        self.existing_id = existing_id


class Conflict(AppError):
    status_code = 409
    code = "conflict"
    default_message = "The request conflicts with the current state."


class EmbeddingMismatch(Conflict):
    code = "embedding_model_mismatch"
    default_message = (
        "The embedding model changed since the documents were indexed. "
        "Re-upload the documents or restore the previous EMBED_MODEL."
    )


class FileTooLarge(AppError):
    status_code = 413
    code = "file_too_large"
    default_message = "The file is larger than the allowed size."


class UnsupportedMediaType(AppError):
    status_code = 415
    code = "unsupported_media_type"
    default_message = "Only PDF files are accepted."


class UnprocessableDocument(AppError):
    status_code = 422
    code = "unprocessable_document"
    default_message = "The document could not be processed."


class CorruptPdf(UnprocessableDocument):
    code = "corrupt_pdf"
    default_message = "The PDF is corrupt or unreadable."


class EncryptedPdf(UnprocessableDocument):
    code = "encrypted_pdf"
    default_message = "Encrypted or password-protected PDFs are not supported."


class ScannedPdf(UnprocessableDocument):
    code = "scanned_pdf"
    default_message = "This PDF has no extractable text (it looks scanned). OCR is not enabled in this build."


class TooManyPages(UnprocessableDocument):
    code = "too_many_pages"
    default_message = "The PDF has more pages than the allowed limit."


class RateLimited(AppError):
    status_code = 429
    code = "rate_limited"
    default_message = "Too many requests. Please slow down."


class OllamaUnavailable(AppError):
    status_code = 503
    code = "llm_unavailable"
    default_message = "The local model server (Ollama) is not reachable. Is it running?"


class LLMTimeout(AppError):
    status_code = 504
    code = "llm_timeout"
    default_message = "The language model took too long to respond."
