#!/usr/bin/env python
import aicrowd_api
import os

########################################################################
# Instatiate Event Notifier
########################################################################
aicrowd_events = aicrowd_api.events.AIcrowdEvents()


def execution_start():
    ########################################################################
    # Register Evaluation Start event
    ########################################################################
    aicrowd_events.register_event(
                event_type=aicrowd_events.AICROWD_EVENT_INFO,
                message="execution_started",
                payload={
                    "event_type": "airborne_detection:execution_started"
                    }
                )

def execution_running():
    ########################################################################
    # Register Evaluation Start event
    ########################################################################
    aicrowd_events.register_event(
                event_type=aicrowd_events.AICROWD_EVENT_INFO,
                message="execution_progress",
                payload={
                    "event_type": "airborne_detection:execution_progress",
                    "progress": 0.0
                    }
                )


def execution_progress(progress):
    ########################################################################
    # Register Evaluation Progress event
    ########################################################################
    aicrowd_events.register_event(
                event_type=aicrowd_events.AICROWD_EVENT_INFO,
                message="execution_progress",
                payload={
                    "event_type": "airborne_detection:execution_progress",
                    "progress" : progress
                    }
                )

def execution_success():
    ########################################################################
    # Register Evaluation Complete event
    ########################################################################
    predictions_output_path = os.getenv("PREDICTIONS_OUTPUT_PATH", False)

    aicrowd_events.register_event(
                event_type=aicrowd_events.AICROWD_EVENT_SUCCESS,
                message="execution_success",
                payload={
                    "event_type": "airborne_detection:execution_success",
                    "predictions_output_path" : predictions_output_path
                    },
                blocking=True
                )

def execution_error(error):
    ########################################################################
    # Register Evaluation Complete event
    ########################################################################
    aicrowd_events.register_event(
                event_type=aicrowd_events.AICROWD_EVENT_ERROR,
                message="execution_error",
                payload={ #Arbitrary Payload
                    "event_type": "airborne_detection:execution_error",
                    "error" : error
                    },
                blocking=True
                )

def is_grading():
    return os.getenv("AICROWD_IS_GRADING", False)
