import logging, json, sys, time, uuid

class JsonFormatter(logging.Formatter):
    def format(self, record):
        payload = {
            "ts": int(time.time() * 1000),
            "level": record.levelname,
            "msg": record.getMessage(),
        }
        if hasattr(record, "extra"):
            # Merge user-provided structured fields
            for k, v in record.extra.items():
                payload[k] = v
        return json.dumps(payload, ensure_ascii=False)

def configure_logging():
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(JsonFormatter())
    root = logging.getLogger()
    root.setLevel(logging.INFO)
    root.handlers = [handler]
    return root

def with_fields(**kwargs):
    rec = logging.LogRecord("logger", logging.INFO, "", 0, "", None, None)
    rec.extra = kwargs
    return rec
