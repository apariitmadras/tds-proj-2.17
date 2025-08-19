import logging, json, sys, time, os

class JSONHandler(logging.StreamHandler):
    def emit(self, record):
        try:
            payload = {
                "ts": time.time(),
                "level": record.levelname,
                "event": getattr(record, "event", record.getMessage()),
                "msg": record.getMessage(),
            }
            if hasattr(record, "extra_fields"): payload.update(record.extra_fields)
            sys.stdout.write(json.dumps(payload) + "\n")
        except Exception:
            super().emit(record)

def configure_logging():
    root = logging.getLogger()
    root.setLevel(os.getenv("LOG_LEVEL", "INFO"))
    for h in list(root.handlers): root.removeHandler(h)
    root.addHandler(JSONHandler())
    return root

def with_fields(event: str, **fields):
    rec = logging.LogRecord(name="app", level=logging.INFO, pathname=__file__,
                            lineno=0, msg=event, args=(), exc_info=None)
    rec.event = event
    rec.extra_fields = fields
    return rec
