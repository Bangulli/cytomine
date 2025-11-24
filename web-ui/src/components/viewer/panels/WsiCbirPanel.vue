<template>
  <div>
    <div class="wsi-retrieval">
      <h1>{{ $t('wsi-retrieval') }}</h1>
      <a @click="closeRetrieval()">
        <span class="fas fa-times-circle"></span>
      </a>
    </div>


    <div>
        <h5>{{ $t('similar-images') }}</h5>
        <button class="button is-small is-fullwidth" @click="$emit('searchSimilarImages')">
            {{ $t('search-similar-image') }}
        </button>
    </div>
  </div>
</template>


<style lang="scss" scoped>
$backgroundPanel: #f2f2f2;

.wsi-retrieval {
  display: flex;
}

h1 {
  margin: 0;
  text-align: center;
}

a {
  margin-left: auto;
}

.table {
  background-color: $backgroundPanel;
  margin-bottom: 1em !important;
  font-size: 0.9em;
}

.table thead tr {
  display: block;
}

.table tbody {
  display: block;
  overflow-x: scroll;
  overflow-y: scroll;
  min-width: 30em;
  max-width: 60em;
  min-height: 30em;
  max-height: 30em;
}

.table th {
  width: 1%;
}
</style>
