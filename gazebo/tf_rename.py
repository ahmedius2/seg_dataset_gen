import sys
import rosbag2_py
from rclpy.serialization import serialize_message, deserialize_message
from rosidl_runtime_py.utilities import get_message

def rename_tf_frame(input_uri, output_uri, old_frame, new_frame):
    old_variants = {old_frame, '/' + old_frame.lstrip('/')}
    new_name = new_frame.lstrip('/')

    # --- Reader setup ---
    reader = rosbag2_py.SequentialReader()
    reader.open(
        rosbag2_py.StorageOptions(uri=input_uri, storage_id='mcap'),
        rosbag2_py.ConverterOptions('cdr', 'cdr')
    )

    # --- Writer setup ---
    writer = rosbag2_py.SequentialWriter()
    writer.open(
        rosbag2_py.StorageOptions(uri=output_uri, storage_id='mcap'),
        rosbag2_py.ConverterOptions('cdr', 'cdr')
    )

    # Register all topics from the input bag into the writer
    type_map = {}
    for topic_meta in reader.get_all_topics_and_types():
        type_map[topic_meta.name] = topic_meta.type
        writer.create_topic(topic_meta)

    renamed_count = 0
    while reader.has_next():
        topic, serialized_data, timestamp = reader.read_next()

        if topic in ('/tf', '/tf_throttled'):
            msg_type = get_message(type_map[topic])
            msg = deserialize_message(serialized_data, msg_type)

            for transform in msg.transforms:
                if transform.header.frame_id in old_variants:
                    transform.header.frame_id = new_name
                    renamed_count += 1
                if transform.child_frame_id in old_variants:
                    transform.child_frame_id = new_name
                    renamed_count += 1

            serialized_data = serialize_message(msg)

        writer.write(topic, serialized_data, timestamp)

    writer.close()
    print(f"Renamed {renamed_count} frame references.")

if __name__ == '__main__':
    rename_tf_frame(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4])
