<script setup lang="ts">
import type { User } from "~/types";
const props = defineProps<{
  user: User;
  endpoint: string;
}>();
const emit = defineEmits(["upload"]);
const dropzoneRef = ref<HTMLElement | null>(null);
const onDrop = async (files: File[] | null, event: DragEvent) => {
  if (!files) return;
  event.preventDefault();
  const promises = files.map(async (f) => {
    await addFile(f);
  });
  await Promise.all(promises);
};
const { isOverDropZone } = useDropZone(dropzoneRef, {
  onDrop,
});
const addFile = async (f: File) => {
  try {
    const formData = new FormData();
    formData.append("file", f);
    const { data:url } =  await useFetch(`/api/${props.endpoint}/${props.user.sub}`, {
      method: "POST",
      body: formData
    }).text();
    emit("upload", unref(url));
  } catch (e) {
    console.log(e);
  }
};
const inputFiles = () => {
  const input = document.createElement("input");
  input.type = "file";
  input.accept = "*/*";
  input.multiple = true;
  input.onchange = async (e) => {
    //@ts-ignore
    const files = e!.target.files as File[];
    if (!files) return;
    const promises = Array.from(files).map(async (f) => {
      await addFile(f);
    });
    await Promise.all(promises);
  };
  input.click();
};
</script>
<template>
  <input :multiple="true" type="file" accept="*/*" class="hidden" />
  <section class="overflow-auto w-full" @click="inputFiles">
    <div class="backdrop-blur-md" ref="dropzoneRef">
      <div
        class="px-12 py-4 min-w-128 max-w-256 cursor-pointer row center"
        :class="isOverDropZone ? 'borded-dashed' : 'border-none'"
      >
      Upload Files <Icon icon="mdi-upload" class="x2" />
       </div>
    </div>
  
  </section>
</template>